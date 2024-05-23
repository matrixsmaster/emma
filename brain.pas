unit brain;
// The Transformer
// I tried to leave as many comments from the OG code as possible

interface

uses
  SysUtils, Math, Dialogs;

type
  PVNAConfig = ^TVNAConfig;
  TVNAConfig = record
    dim: integer; // transformer dimension
    hidden_dim: integer; // for ffn layers
    n_layers: integer; // number of layers
    n_heads: integer; // number of query heads
    n_kv_heads: integer; // number of key/value heads (can be < query heads because of multiquery)
    vocab_size: integer; // vocabulary size, usually 256 (byte-level)
    seq_len: integer; // max sequence length
    shared_classifier: integer;
    GS: integer;
    data_size: cardinal; // uint32
  end;

  PInt8 = array of ShortInt; // int8
  PFloats = array of single;

  TQuantizedTensor = record
    q: PInt8;
    s: PFloats;
  end;

  PTransformerWeights = ^TTransformerWeights;
  TTransformerWeights = record
    // token embedding table
    q_tokens: TQuantizedTensor; // (vocab_size, dim)
    token_embedding_table: PFloats; // (vocab_size, dim)
    // weights for rmsnorms
    rms_att_weight: PFloats; // (layer, dim) rmsnorm weights
    rms_ffn_weight: PFloats; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    wq: TQuantizedTensor; // (layer, dim, n_heads * head_size)
    wk: TQuantizedTensor; // (layer, dim, n_kv_heads * head_size)
    wv: TQuantizedTensor; // (layer, dim, n_kv_heads * head_size)
    wo: TQuantizedTensor; // (layer, n_heads * head_size, dim)
    // weights for ffn
    w1: TQuantizedTensor; // (layer, hidden_dim, dim)
    w2: TQuantizedTensor; // (layer, dim, hidden_dim)
    w3: TQuantizedTensor; // (layer, hidden_dim, dim)
    // final rmsnorm
    rms_final_weight: PFloats; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    wcls: TQuantizedTensor;
  end;

  PRunState = ^TRunState;
  TRunState = record
    // current wave of activations
    x: PFloats; // activation at current time stamp (dim,)
    xb: PFloats; // same, but inside a residual branch (dim,)
    xb2: PFloats; // an additional buffer just for convenience (dim,)
    hb: PFloats; // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: PFloats; // buffer for hidden dimension in the ffn (hidden_dim,)
    xq: TQuantizedTensor; // quantized x (dim,)
    hq: TQuantizedTensor; // quantized hb (hidden_dim,)
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
    config: TVNAConfig; // the hyperparameters of the architecture (the blueprint)
    weights: TTransformerWeights; // the weights of the model
    state: TRunState; // buffers for the "wave" of activations in the forward pass
  end;

  EFileReadError = class(Exception);
  TLoadCallbkFun = function(total, cur: integer): boolean of object;

const
  IOBufSize = 10000;
  epsilon = 1.0e-5;
  Q_MAX = 127.0;

var
  GS: integer;
  brain_err: string;

procedure FileReadWithFail(Handle: Integer; var Buffer; Count: Integer); forward;
procedure build_transformer(fhandle: integer; t: PTransformer; callback: TLoadCallbkFun); forward;
procedure free_transformer(t: PTransformer); forward;
procedure softmax(var x: PFloats; off, size: integer); forward;
function nn_ff(transformer: PTransformer; token, pos: integer): PFloats; forward;
procedure dequantize(qx: TQuantizedTensor; var x: PFloats; n: integer); forward;
procedure quantize(qx: TQuantizedTensor; var x: PFloats; n: integer); forward;

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

procedure qtensor_copy(var o: TQuantizedTensor; inp: TQuantizedTensor);
var
  i: integer;
begin
  SetLength(o.q,High(inp.q)+1);
  SetLength(o.s,High(inp.s)+1);
  for i := 0 to High(inp.q) do o.q[i] := inp.q[i];
  for i := 0 to High(inp.s) do o.s[i] := inp.s[i];
end;

procedure FileReadWithFail(Handle: Integer; var Buffer; Count: Integer);
begin
  if FileRead(Handle,Buffer,Count) <> Count then
    raise EFileReadError.Create('File read failed: unable to read '+IntToStr(Count)+' bytes @ 0x' + IntToHex(FileSeek(Handle,0,1),10));
end;

procedure malloc_run_state(s: PRunState; p: PVNAConfig);
var
  kv_dim: integer;
begin
  kv_dim := round((p^.dim * p^.n_kv_heads) / p^.n_heads);

  // allocate float tensors
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

  // allocate quantized runtime tensors
  SetLength(s^.xq.q,p^.dim);
  SetLength(s^.xq.s,p^.dim); // shouldn't it be "div GS" ?
  SetLength(s^.hq.q,p^.hidden_dim);
  SetLength(s^.hq.s,p^.hidden_dim);
end;

procedure free_tensor(var z: TQuantizedTensor);
begin
  z.q := nil;
  z.s := nil;
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
  free_tensor(s^.xq);
  free_tensor(s^.hq);
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

procedure qalloc_and_qread(var arr: TQuantizedTensor; fhandle: integer; amount, size_each: integer; callback: TLoadCallbkFun);
var
  i,j: Integer;
  nq,ns,size,r: cardinal;
  a: array[0..IOBufSize] of ShortInt;
  f: array[0..IOBufSize] of single;
begin
  SetLength(arr.q,amount*size_each);
  SetLength(arr.s,amount*(size_each div GS));
  nq := 0;
  ns := 0;
  for i := 0 to amount-1 do
  begin
    // read the quantized weights
    size := size_each;
    while size > 0 do
    begin
      if size > IOBufSize then r := IOBufSize
      else r := size;
      FileReadWithFail(fhandle,a,r*sizeof(ShortInt));
      size := size - r;
      for j := 0 to r-1 do
      begin
        arr.q[nq] := a[j];
        inc(nq);
      end;
    end;

    // read the scale factors
    size := size_each div GS;
    while size > 0 do
    begin
      if size > IOBufSize then r := IOBufSize
      else r := size;
      FileReadWithFail(fhandle,f,r*sizeof(single));
      size := size - r;
      for j := 0 to r-1 do
      begin
        arr.s[ns] := f[j];
        inc(ns);
      end;
    end;

    // callback for interactivity
    if @callback <> nil then
    begin
      if not callback(amount,i+1) then
        raise EFileReadError.Create('Aborted by user');
    end;
  end;
end;

procedure load_weights(fhandle: integer; trans: PTransformer; callback: TLoadCallbkFun);
var
  p: PVNAConfig;
  w: PTransformerWeights;
  head_size: integer;
  tokz: cardinal;
begin
  p := @trans^.config;
  w := @trans^.weights;
  head_size := p^.dim div p^.n_heads;

  // no comments, it's a mess. Delphi doesn't have proper pointer arithmetic, so... improvisation
  alloc_and_read(w^.rms_att_weight,fhandle,p^.n_layers * p^.dim,callback);
  alloc_and_read(w^.rms_ffn_weight,fhandle,p^.n_layers * p^.dim,callback);
  alloc_and_read(w^.rms_final_weight,fhandle,p^.dim,callback);

  tokz := p^.vocab_size * p^.dim;
  qalloc_and_qread(w^.q_tokens,fhandle,1,tokz,callback);
  SetLength(w^.token_embedding_table,tokz);
  dequantize(w^.q_tokens,w^.token_embedding_table,tokz);

  qalloc_and_qread(w^.wq,fhandle,p^.n_layers,p^.dim * (p^.n_heads * head_size),callback);
  qalloc_and_qread(w^.wk,fhandle,p^.n_layers,p^.dim * (p^.n_kv_heads * head_size),callback);
  qalloc_and_qread(w^.wv,fhandle,p^.n_layers,p^.dim * (p^.n_kv_heads * head_size),callback);
  qalloc_and_qread(w^.wo,fhandle,p^.n_layers,(p^.n_heads * head_size) * p^.dim,callback);

  qalloc_and_qread(w^.w1,fhandle,p^.n_layers,p^.dim * p^.hidden_dim,callback);
  qalloc_and_qread(w^.w2,fhandle,p^.n_layers,p^.hidden_dim * p^.dim,callback);
  ShowMessage(IntToStr(FileSeek(fhandle,0,1)));
  qalloc_and_qread(w^.w3,fhandle,p^.n_layers,p^.dim * p^.hidden_dim,callback);
  ShowMessage(IntToStr(FileSeek(fhandle,0,1)));

  if trans^.config.shared_classifier > 0 then
    qtensor_copy(w^.wcls,w^.q_tokens)
  else
    qalloc_and_qread(w^.wcls,fhandle,1,p^.vocab_size * p^.dim,callback);
end;

procedure free_weights(w: PTransformerWeights);
begin
  free_tensor(w^.q_tokens);
  w^.token_embedding_table := nil;
  w^.rms_att_weight := nil;
  w^.rms_ffn_weight := nil;
  free_tensor(w^.wq);
  free_tensor(w^.wk);
  free_tensor(w^.wv);
  free_tensor(w^.wo);
  free_tensor(w^.w1);
  free_tensor(w^.w2);
  free_tensor(w^.w3);
  w^.rms_final_weight := nil;
  free_tensor(w^.wcls);
end;

procedure build_transformer(fhandle: integer; t: PTransformer; callback: TLoadCallbkFun);
begin
  FileReadWithFail(fhandle,t^.config,sizeof(TVNAConfig));
  GS := t^.config.GS;
  load_weights(fhandle,t,callback);
  malloc_run_state(@t^.state, @t^.config);
end;

procedure free_transformer(t: PTransformer);
begin
  free_run_state(@t^.state);
  free_weights(@t^.weights);
end;

// *****************************************************************************
// Mmmm... Mathematics :)
// A word of advice - don't try to follow the offset-based arithmetic here -
// you'll lose your mind. You've been warned! ;)

procedure dequantize(qx: TQuantizedTensor; var x: PFloats; n: integer);
var
  i: integer;
begin
  for i := 0 to n-1 do
    x[i] := qx.q[i] * qx.s[i div GS];
end;

procedure quantize(qx: TQuantizedTensor; var x: PFloats; n: integer);
var
  i,num_groups,group: integer;
  wmax,val,scale,quant_value: single;
  quantized: ShortInt;
begin
  num_groups := n div GS;
  for group := 0 to num_groups-1 do
  begin
    // find the max absolute value in the current group
    wmax := 0.0;
    for i := 0 to GS-1 do
    begin
      val := abs(x[group * GS + i]);
      if val > wmax then wmax := val;
    end;

    // calculate and write the scaling factor
    scale := wmax / Q_MAX;
    qx.s[group] := scale;

    // calculate and write the quantized values
    for i := 0 to GS-1 do
    begin
      quant_value := x[group * GS + i] / scale; // scale
      quantized := round(quant_value); // round and clamp
      qx.q[group * GS + i] := quantized;
    end;
  end;
end;

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

procedure matmul(var xout: PFloats; x,w: TQuantizedTensor; woff,n,d: integer);
var
  i,j,k: integer;
  val: single;
  ival,inoff,qa,qb: integer;
begin
  // W (d,n) @ x (n,) -> xout (d,)
  // by far the most amount of time is spent inside this little function
  for i := 0 to d-1 do
  begin
    val := 0.0;
    ival := 0;
    inoff := woff + i * n;
    j := 0;
    while j <= n - GS do
    begin
      for k := 0 to GS-1 do
      begin
        qa := x.q[j + k];
        qb := w.q[inoff + j + k];
        ival := ival + (qa * qb);
      end;
      val := val + ival * w.s[(inoff + j) div GS] * x.s[j div GS];
      ival := 0;
      inc(j,GS);
    end;
    xout[i] := val;
  end;
end;

// *****************************************************************************
// Feed forward with attention (aka a Transformer)

function nn_ff(transformer: PTransformer; token, pos: integer): PFloats;
var
  i,j,l,h,ts: integer;
  p: PVNAConfig;
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
  floatcpy_off(x,w^.token_embedding_table,token*dim,dim);

  // forward all the layers
  for l := 0 to p^.n_layers-1 do
  begin
    // attention rmsnorm
    rmsnorm(s^.xb, x, w^.rms_att_weight, l*dim, dim);

    // qkv matmuls for this position
    quantize(s^.xq, s^.xb, dim);
    matmul(s^.q, s^.xq, w^.wq, l*dim*dim, dim, dim);
    matmul(s^.k, s^.xq, w^.wk, l*dim*kv_dim, dim, kv_dim);
    matmul(s^.v, s^.xq, w^.wv, l*dim*kv_dim, dim, kv_dim);

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
    floatcpy_doff(s^.key_cache, loff+pos*kv_dim, s^.k, 0, kv_dim);
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
    quantize(s^.xq, s^.xb, dim);
    matmul(s^.xb2, s^.xq, w^.wo, l*dim*dim, dim, dim);

    // residual connection back into x
    for i := 0 to dim-1 do x[i] := x[i] + s^.xb2[i];

    // ffn rmsnorm
    rmsnorm(s^.xb, x, w^.rms_ffn_weight, l*dim, dim);

    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
    quantize(s^.xq, s^.xb, dim);
    matmul(s^.hb, s^.xq, w^.w1, l*dim*hidden_dim, dim, hidden_dim);
    matmul(s^.hb2, s^.xq, w^.w3, l*dim*hidden_dim, dim, hidden_dim);

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
    quantize(s^.hq, s^.hb, dim);
    matmul(s^.xb, s^.hq, w^.w2, l*dim*hidden_dim, hidden_dim, dim);

    // residual connection
    for i := 0 to dim-1 do x[i] := x[i] + s^.xb[i];
  end;

  // final rmsnorm
  rmsnorm(x, x, w^.rms_final_weight, 0, dim);

  // classifier into logits
  quantize(s^.xq, s^.x, dim);
  matmul(s^.logits, s^.xq, w^.wcls, 0, p^.dim, p^.vocab_size);
  Result := s^.logits;
end;

end.
