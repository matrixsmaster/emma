unit sampler;
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

interface

uses
  SysUtils, Math, Classes, brain;

type
  PTProbIndex = ^TProbIndex;
  TProbIndex = record // struct used when sorting probabilities during top-p sampling
    prob: single;
    index: integer;
  end;
  TProbIndexArr = array of TProbIndex;

  PSampler = ^TSampler;
  TSampler = record
    vocab_size: integer;
    probindex: TProbIndexArr; // buffer used in top-p sampling
    temperature: single;
    topp: single;
    rng_state: Int64;
  end;

procedure build_sampler(sampler: PSampler; temperature, topp: single; vocab_size, rng_seed: integer); forward;
procedure free_sampler(sampler: PSampler); forward;
function random_f32(var state: Int64): single; forward;
function sample(sampler: PSampler; var logits: PFloats): integer; forward;

implementation

function random_u32(var state: Int64): cardinal;
begin
  state := state xor (state shr 12);
  state := state xor (state shl 25);
  state := state xor (state shr 27);
  Result := (state * Int64($2545F4914F6CDD1D)) shr 32;
end;

// random float32 in [0,1)
function random_f32(var state: Int64): single;
begin
  Result := (random_u32(state) shr 8) / 16777216.0;
end;

procedure build_sampler(sampler: PSampler; temperature, topp: single; vocab_size, rng_seed: integer);
begin
  sampler^.vocab_size := vocab_size;
  sampler^.temperature := temperature;
  sampler^.topp := topp;
  sampler^.rng_state := rng_seed;
  // buffer only used with nucleus sampling; may not need but it's ~small
  SetLength(sampler^.probindex,vocab_size);
end;

procedure free_sampler(sampler: PSampler);
begin
  sampler^.probindex := nil;
end;

function sample_argmax(probabilities: PFloats; n: integer): integer;
var
  i,max_i: integer;
  max_p: single;
begin
  // return the index that has the highest probability
  max_i := 0;
  max_p := probabilities[0];
  for i := 1 to n-1 do
  begin
    if (probabilities[i] > max_p) then
    begin
      max_i := i;
      max_p := probabilities[i];
    end;
  end;
  Result := max_i;
end;

function sample_mult(probabilities: PFloats; n: integer; coin: single): integer;
var
  i: integer;
  cdf: single;
begin
  // sample index from probabilities (they must sum to 1!)
  // coin is a random number in [0, 1), usually from random_f32()
  cdf := 0;
  for i := 0 to n-1 do
  begin
    cdf := cdf + probabilities[i];
    if coin < cdf then
    begin
      Result := i;
      exit;
    end;
  end;
  Result := n - 1; // in case of rounding errors
end;

function probs_compare(a,b: pointer): integer;
var
  ia,ib: PTProbIndex;
begin
  ia := a;
  ib := b;
  if ia^.prob = ib^.prob then Result := 0
  else if ia^.prob < ib^.prob then Result := 1
  else Result := -1;
end;

function sample_topp(probabilities: PFloats; n: integer; topp, coin: single; var probindex: TProbIndexArr): integer;
var
  i,n0: integer;
  cutoff,cumulative_prob: single;
  tmpprobs: TProbIndexArr;
  tmplst: TList;
  last_idx: integer;
  r,cdf: single;
begin
  // top-p sampling (or "nucleus sampling") samples from the smallest set of
  // tokens that exceed probability topp. This way we never sample tokens that
  // have very low probabilities and are less likely to go "off the rails".
  // coin is a random number in [0, 1), usually from random_f32()

  // sort indices in descending order of probabilities
  // values smaller than (1 - topp) / (n - 1) cannot be part of the result
  // so for efficiency we crop these out as candidates before sorting
  n0 := 0;
  cutoff := (1.0 - topp) / (n - 1);
  SetLength(tmpprobs,n);
  tmplst := TList.Create;
  for i := 0 to n-1 do
  begin
    if probabilities[i] >= cutoff then
    begin
      tmpprobs[n0].index := i;
      tmpprobs[n0].prob := probabilities[i];
      tmplst.Add(@(tmpprobs[n0]));
      inc(n0);
    end;
  end;
  tmplst.Sort(probs_compare);

  for i := 0 to tmplst.Count-1 do
    probindex[i] := PTProbIndex(tmplst[i])^;
  tmplst.Free;
  tmpprobs := nil;

  // truncate the list where cumulative probability exceeds topp
  cumulative_prob := 0;
  last_idx := n0 - 1; // in case of rounding errors consider all elements
  for i := 0 to n0-1 do
  begin
    cumulative_prob := cumulative_prob + probindex[i].prob;
    if cumulative_prob > topp then
    begin
      last_idx := i;
      break; // we've exceeded topp by including last_idx
    end;
  end;

  // sample from the truncated list
  r := coin * cumulative_prob;
  cdf := 0;
  for i := 0 to last_idx do
  begin
    cdf := cdf + probindex[i].prob;
    if r < cdf then
    begin
      Result := probindex[i].index;
      exit;
    end;
  end;
  Result := probindex[last_idx].index; // in case of rounding errors
end;

function sample(sampler: PSampler; var logits: PFloats): integer;
var
  next,q: integer;
  coin: single;
begin
  // sample the token given the logits and some hyperparameters
  if sampler^.temperature = 0 then
  begin
    // greedy argmax sampling: take the token with the highest probability
    next := sample_argmax(logits, sampler^.vocab_size);
  end
  else
  begin
    // apply the temperature to the logits
    for q := 0 to sampler^.vocab_size-1 do
      logits[q] := logits[q] / sampler^.temperature;
    // apply softmax to the logits to get the probabilities for next token
    softmax(logits, 0, sampler^.vocab_size);
    // flip a (float) coin (this is our source of entropy for sampling)
    coin := random_f32(sampler^.rng_state);
    // we sample from this distribution to get the next token
    if (sampler^.topp <= 0) or (sampler^.topp >= 1) then
    begin
      // simply sample from the predicted probability distribution
      next := sample_mult(logits, sampler^.vocab_size, coin);
    end
    else
    begin
      // top-p (nucleus) sampling, clamping the least likely tokens to zero
      next := sample_topp(logits, sampler^.vocab_size, sampler^.topp, coin, sampler^.probindex);
    end;
  end;
  Result := next;
end;

end.
 