unit brain;
// The Transformer

interface

uses
  SysUtils, Math;

type
  PConfig = ^TConfig;
  TConfig = record
    dim: integer; // transformer dimension
    hidden_dim: integer; // for ffn layers
    n_layers: integer; // number of layers
    n_heads: integer; // number of query heads
    n_kv_heads: integer; // number of key/value heads (can be < query heads because of multiquery)
    vocab_size: integer; // vocabulary size, usually 256 (byte-level)
    seq_len: integer; // max sequence length
  end;

  PFloats = array of single;

  PTransformerWeights = ^TTransformerWeights;
  TTransformerWeights = record
    // token embedding table
    token_embedding_table: PFloats;    // (vocab_size, dim)
    // weights for rmsnorms
    rms_att_weight: PFloats; // (layer, dim) rmsnorm weights
    rms_ffn_weight: PFloats; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    wq: PFloats; // (layer, dim, n_heads * head_size)
    wk: PFloats; // (layer, dim, n_kv_heads * head_size)
    wv: PFloats; // (layer, dim, n_kv_heads * head_size)
    wo: PFloats; // (layer, n_heads * head_size, dim)
    // weights for ffn
    w1: PFloats; // (layer, hidden_dim, dim)
    w2: PFloats; // (layer, dim, hidden_dim)
    w3: PFloats; // (layer, hidden_dim, dim)
    // final rmsnorm
    rms_final_weight: PFloats; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    wcls: PFloats;
  end;

  PRunState = ^TRunState;
  TRunState = record
    // current wave of activations
    x: PFloats; // activation at current time stamp (dim,)
    xb: PFloats; // same, but inside a residual branch (dim,)
    xb2: PFloats; // an additional buffer just for convenience (dim,)
    hb: PFloats; // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: PFloats; // buffer for hidden dimension in the ffn (hidden_dim,)
    q: PFloats; // query (dim,)
    k: PFloats; // key (dim,)
    v: PFloats; // value (dim,)
    att: PFloats; // buffer for scores/attention values (n_heads, seq_len)
    logits: PFloats; // output logits
    // kv cache
    key_cache: PFloats;   // (layer, seq_len, dim)
    value_cache: PFloats; // (layer, seq_len, dim)
  end;

  PTransformer = ^TTransformer;
  TTransformer = record
    config: TConfig; // the hyperparameters of the architecture (the blueprint)
    weights: TTransformerWeights; // the weights of the model
    state: TRunState; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    //fd: integer; // file descriptor for memory mapping
    //data: PFloats; // memory mapped data pointer
    //file_size: cardinal; // size of the checkpodata in bytes: integer
  end;

  EFileReadError = class(Exception);
  TLoadCallbkFun = function(total, cur: integer): boolean of object;

const
  IOBufSize = 10000;
  epsilon = 1.0e-5;

var
  brain_err: string;

procedure FileReadWithFail(Handle: Integer; var Buffer; Count: Integer); forward;
procedure build_transformer(fhandle: integer; t: PTransformer; callback: TLoadCallbkFun); forward;
procedure free_transformer(t: PTransformer); forward;
procedure softmax(var x: PFloats; off, size: integer); forward;
function nn_ff(transformer: PTransformer; token, pos: integer): PFloats; forward;

implementation

// *****************************************************************************
// I/O stuff for loading weights

procedure floatcpy_off(var o: PFloats; inp: PFloats; inp_off, size: integer);
var
  i: integer;
begin
  for i := 0 to size-1 do o[i] := inp[i+inp_off];
end;

procedure floatcpy_doff(var o: PFloats; out_off: integer; inp: PFloats; inp_off, size: integer);
var
  i: integer;
begin
  for i := 0 to size-1 do o[i+out_off] := inp[i+inp_off];
end;

procedure FileReadWithFail(Handle: Integer; var Buffer; Count: Integer);
begin
  if FileRead(Handle,Buffer,Count) <> Count then
    raise EFileReadError.Create('File read failed: unable to read '+IntToStr(Count)+' bytes');
end;

procedure malloc_run_state(s: PRunState; p: PConfig);
var
  kv_dim: integer;
begin
  kv_dim := round((p^.dim * p^.n_kv_heads) / p^.n_heads);
  SetLength(s^.x,p^.dim);
  SetLength(s^.xb,p^.dim);
  SetLength(s^.xb2,p^.dim);
  SetLength(s^.hb,p^.hidden_dim);
  SetLength(s^.hb2,p^.hidden_dim);
  SetLength(s^.q,p^.dim);
  SetLength(s^.k,kv_dim);
  SetLength(s^.v,kv_dim);
  SetLength(s^.att,p^.n_heads * p^.seq_len);
  SetLength(s^.logits,p^.vocab_size);
  SetLength(s^.key_cache,p^.n_layers * p^.seq_len * kv_dim);
  SetLength(s^.value_cache,p^.n_layers * p^.seq_len * kv_dim);
end;

procedure free_run_state(s: PRunState);
begin
  s^.x := nil;
  s^.xb := nil;
  s^.xb2 := nil;
  s^.hb := nil;
  s^.hb2 := nil;
  s^.q := nil;
  s^.k := nil;
  s^.v := nil;
  s^.att := nil;
  s^.logits := nil;
  s^.key_cache := nil;
  s^.value_cache := nil;
end;

procedure alloc_and_read(var arr: PFloats; fhandle: integer; size: cardinal; callback: TLoadCallbkFun);
var
  r,j,p,ogs: cardinal;
  a: array[0..IOBufSize] of single;
begin
  SetLength(arr,size);
  // The following code is because Delphi doesn't have proper pointer arithmetic
  // and its IO code is just as janky as BCB's one (go figure, right?) :D
  p := 0;
  ogs := size;
  while size > 0 do
  begin
    if size > IOBufSize then r := IOBufSize
    else r := size;
    // read the data
    FileReadWithFail(fhandle,a,r*sizeof(single));
    size := size - r;
    // and copy it to the actual recepient array
    for j := 0 to r-1 do
    begin
      arr[p] := a[j];
      inc(p);
    end;

    // callback time (retrofitted into this function)
    if @callback <> nil then
    begin
      if not callback(ogs,ogs-size) then
        raise EFileReadError.Create('Aborted by user');
    end;
  end;
end;

procedure load_weights(fhandle: integer; w: PTransformerWeights; p: PConfig; shared_weights: boolean; callback: TLoadCallbkFun);
var
  head_size: integer;
begin
  head_size := p^.dim div p^.n_heads;

  alloc_and_read(w^.token_embedding_table,fhandle,p^.vocab_size * p^.dim,callback);
  alloc_and_read(w^.rms_att_weight,fhandle,p^.n_layers * p^.dim,callback);
  alloc_and_read(w^.wq,fhandle,p^.n_layers * p^.dim * (p^.n_heads * head_size),callback);
  alloc_and_read(w^.wk,fhandle,p^.n_layers * p^.dim * (p^.n_kv_heads * head_size),callback);
  alloc_and_read(w^.wv,fhandle,p^.n_layers * p^.dim * (p^.n_kv_heads * head_size),callback);
  alloc_and_read(w^.wo,fhandle,p^.n_layers * (p^.n_heads * head_size) * p^.dim,callback);
  alloc_and_read(w^.rms_ffn_weight,fhandle,p^.n_layers * p^.dim,callback);
  alloc_and_read(w^.w1,fhandle,p^.n_layers * p^.dim * p^.hidden_dim,callback);
  alloc_and_read(w^.w2,fhandle,p^.n_layers * p^.hidden_dim * p^.dim,callback);
  alloc_and_read(w^.w3,fhandle,p^.n_layers * p^.dim * p^.hidden_dim,callback);
  alloc_and_read(w^.rms_final_weight,fhandle,p^.dim,callback);

  FileSeek(fhandle,p^.seq_len * head_size div 2 * sizeof(single),1); // skip what used to be freq_cis_real (for RoPE)
  FileSeek(fhandle,p^.seq_len * head_size div 2 * sizeof(single),1); // skip what used to be freq_cis_imag (for RoPE)

  if shared_weights then
  begin
    SetLength(w^.wcls,p^.vocab_size * p^.dim);
    floatcpy_off(w^.wcls,w^.token_embedding_table,0,p^.vocab_size * p^.dim);
  end
  else
    alloc_and_read(w^.wcls,fhandle,p^.vocab_size * p^.dim,callback);
end;

procedure free_weights(w: PTransformerWeights);
begin
  w^.token_embedding_table := nil;
  w^.rms_att_weight := nil;
  w^.wq := nil;
  w^.wk := nil;
  w^.wv := nil;
  w^.wo := nil;
  w^.rms_ffn_weight := nil;
  w^.w1 := nil;
  w^.w2 := nil;
  w^.w3 := nil;
  w^.rms_final_weight := nil;
  w^.wcls := nil;
end;

procedure read_checkpoint(fhandle: integer; cfg: PConfig; weights: PTransformerWeights; callback: TLoadCallbkFun);
begin
  FileSeek(fhandle,4,0); // skip data block size
  // read in the config header
  FileReadWithFail(fhandle,cfg^,sizeof(TConfig));
  // negative vocab size is hacky way of signaling unshared weights
  load_weights(fhandle,weights,cfg,(cfg^.vocab_size > 0),callback);
end;

procedure build_transformer(fhandle: integer; t: PTransformer; callback: TLoadCallbkFun);
begin
  // read in the Config and the Weights from the checkpoint
  read_checkpoint(fhandle, @t^.config, @t^.weights,callback);
  // allocate the RunState buffers
  malloc_run_state(@t^.state, @t^.config);
end;

procedure free_transformer(t: PTransformer);
begin
  free_run_state(@t^.state);
  free_weights(@t^.weights);
end;

// *****************************************************************************
// Mathematics :)

procedure rmsnorm(var o: PFloats; x, weight: PFloats; woff, size: integer);
var
  i: integer;
  ss: single;
begin
  // calculate sum of squares
  ss := 0;
  for i := 0 to size-1 do
    ss := ss + x[i] * x[i];
  ss := ss / size + epsilon;
  ss := 1.0 / sqrt(ss);
  // normalize and scale
  for i := 0 to size-1 do
    o[i] := weight[i+woff] * (ss * x[i]);
end;

procedure softmax(var x: PFloats; off, size: integer);
var
  i,osz: integer;
  max_val,sum: single;
begin
  osz := off + size - 1;
  // find max value (for numerical stability)
  max_val := x[off];
  for i := off+1 to osz do
  begin
    if x[i] > max_val then max_val := x[i];
  end;
  // exp and sum
  sum := 0;
  for i := off to osz do
  begin
    x[i] := exp(x[i] - max_val);
    sum := sum + x[i];
  end;
  // normalize
  for i := off to osz do
    x[i] := x[i] / sum;
end;

procedure matmul(var xout: PFloats; x: PFloats; w: PFloats; woff, n, d: integer);
var
  i,j: integer;
  val: single;
begin
  // W (d,n) @ x (n,) -> xout (d,)
  // by far the most amount of time is spent inside this little function
  for i := 0 to d-1 do
  begin
    val := 0;
    for j := 0 to n-1 do val := val + w[i * n + j + woff] * x[j];
    xout[i] := val;
  end;
end;

// *****************************************************************************
// Feed forward with attention (aka a Transformer)

function nn_ff(transformer: PTransformer; token, pos: integer): PFloats;
var
  i,j,l,h,ts: integer;
  p: PConfig;
  w: PTransformerWeights;
  s: PRunState;
  x,vr: PFloats;
  dim,kv_dim,kv_mul,hidden_dim,head_size,head_dim,rotn: integer;
  loff,qoff,atoff,koff,xboff,voff: integer;
  freq,val,fcr,fci,v0,v1,score,a: single;
begin
  // a few convenience variables
  p := @transformer^.config;
  w := @transformer^.weights;
  s := @transformer^.state;
  x := s^.x;
  dim := p^.dim;
  kv_dim := (p^.dim * p^.n_kv_heads) div p^.n_heads;
  kv_mul := p^.n_heads div p^.n_kv_heads; // integer multiplier of the kv sharing in multiquery
  hidden_dim :=  p^.hidden_dim;
  head_size := dim div p^.n_heads;

  // copy the token embedding into x
  //content_row := w^.token_embedding_table + token * dim;
  //memcpy(x, content_row, dim*sizeof( *x));
  floatcpy_off(x,w^.token_embedding_table,token*dim,dim);

  // forward all the layers
  for l := 0 to p^.n_layers-1 do
  begin
    // attention rmsnorm
    rmsnorm(s^.xb, x, w^.rms_att_weight, l*dim, dim);

    // qkv matmuls for this position
    matmul(s^.q, s^.xb, w^.wq, l*dim*dim, dim, dim);
    matmul(s^.k, s^.xb, w^.wk, l*dim*kv_dim, dim, kv_dim);
    matmul(s^.v, s^.xb, w^.wv, l*dim*kv_dim, dim, kv_dim);

    // RoPE relative positional encoding: complex-valued rotate q and k in each head
    i := 0;
    while i < dim do
    begin
      head_dim := i mod head_size;
      freq := 1.0 / power(10000.0, head_dim / head_size);
      val := pos * freq;
      fcr := cos(val);
      fci := sin(val);
      rotn := 1; // how many vectors? 2 = q & k, 1 = q only
      if i < kv_dim then rotn := 2;

      for j := 0 to rotn-1 do
      begin
        // the vector to rotate (query or key)
        vr := s^.k;
        if j = 0 then vr := s^.q;

        v0 := vr[i];
        v1 := vr[i+1];
        vr[i]   := v0 * fcr - v1 * fci;
        vr[i+1] := v0 * fci + v1 * fcr;
      end;
      inc(i,2);
    end;

    // save key,value at this time step (pos) to our kv cache
    loff := l * p^.seq_len * kv_dim; // kv cache layer offset for convenience
    //key_cache_row : PFloats= s^.key_cache + loff + pos * kv_dim;
    //value_cache_row : PFloats= s^.value_cache + loff + pos * kv_dim;
    //memcpy(key_cache_row, s^.k, kv_dim * sizeof( *key_cache_row));
    floatcpy_doff(s^.key_cache, loff+pos*kv_dim, s^.k, 0, kv_dim);
    //memcpy(value_cache_row, s^.v, kv_dim * sizeof( *value_cache_row));
    floatcpy_doff(s^.value_cache, loff+pos*kv_dim, s^.v, 0, kv_dim);

    // multihead attention. iterate over all heads
    for h := 0 to p^.n_heads-1 do
    begin
      // get the query vector for this head
      qoff := h * head_size;
      // attention scores for this head
      atoff := h * p^.seq_len;
      // iterate over all timesteps, including the current one
      for ts := 0 to pos do
      begin
        // get the key vector for this head and at this timestep
        koff := loff + ts * kv_dim + (h div kv_mul) * head_size;
        // calculate the attention score as the dot product of q and k
        score := 0;
        for i := 0 to head_size-1 do
            score := score + s^.q[i+qoff] * s^.key_cache[i+koff];
        score := score / sqrt(head_size);
        // save the score to the attention buffer
        s^.att[ts+atoff] := score;
      end;

      // softmax the scores to get attention weights, from 0..pos inclusively
      softmax(s^.att, atoff, pos + 1);

      // weighted sum of the values, store back into xb
      xboff := h * head_size;
      for i := 0 to head_size-1 do s^.xb[i+xboff] := 0;
      for ts := 0 to pos do
      begin
        // get the value vector for this head and at this timestep
        voff := loff + ts * kv_dim + (h div kv_mul) * head_size;
        // get the attention weight for this timestep
        a := s^.att[ts+atoff];
        // accumulate the weighted value into xb
        for i := 0 to head_size-1 do
            s^.xb[i+xboff] := s^.xb[i+xboff] + a * s^.value_cache[i+voff];
      end;
    end;

    // final matmul to get the output of the attention
    matmul(s^.xb2, s^.xb, w^.wo, l*dim*dim, dim, dim);

    // residual connection back into x
    for i := 0 to dim-1 do x[i] := x[i] + s^.xb2[i];

    // ffn rmsnorm
    rmsnorm(s^.xb, x, w^.rms_ffn_weight, l*dim, dim);

    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
    matmul(s^.hb, s^.xb, w^.w1, l*dim*hidden_dim, dim, hidden_dim);
    matmul(s^.hb2, s^.xb, w^.w3, l*dim*hidden_dim, dim, hidden_dim);

    // SwiGLU non-linearity
    for i := 0 to hidden_dim-1 do
    begin
      val := s^.hb[i];
      // silu(x)=x*sigma(x), where sigma(x) is the logistic sigmoid
      val := val * (1.0 / (1.0 + exp(-val)));
      // elementwise multiply with w3(x)
      val := val * s^.hb2[i];
      s^.hb[i] := val;
    end;

    // final matmul to get the output of the ffn
    matmul(s^.xb, s^.hb, w^.w2, l*dim*hidden_dim, hidden_dim, dim);

    // residual connection
    for i := 0 to dim-1 do x[i] := x[i] + s^.xb[i];
  end;

  // final rmsnorm
  rmsnorm(x, x, w^.rms_final_weight, 0, dim);

  // classifier into logits
  matmul(s^.logits, x, w^.wcls, 0, p^.dim, p^.vocab_size);
  Result := s^.logits;
end;

end.
