unit tokenz;
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

interface

uses
  SysUtils, Math, Classes, brain;

type
  PInts = array of integer;

  PTokenIndex = ^TTokenIndex;
  TTokenIndex = record
    str: string;
    id: integer;
  end;

  PTokenizer = ^TTokenizer;
  TTokenizer = record
    vocab: array of string;
    vocab_scores: PFloats;
    sorted_vocab: array of TTokenIndex;
    vocab_size: integer;
    max_token_length: cardinal;
    //byte_pieces: array[0..511] of string; // stores all single-byte strings
  end;

procedure build_tokenizer(fhandle: integer; t: PTokenizer; vocab_size: integer; callback: TLoadCallbkFun); forward;
procedure free_tokenizer(t: PTokenizer); forward;
function decode(t: PTokenizer; prev_token, token: integer): string; forward;
procedure encode(var res: PInts; t: PTokenizer; text: string; bos, eos: boolean); forward;

implementation

uses StrUtils;

procedure tokenizer_load_from_file(fhandle: integer; t: PTokenizer; vocab_size: integer; callback: TLoadCallbkFun);
var
  i,len: integer;
begin
  FileReadWithFail(fhandle,t^.max_token_length,sizeof(t^.max_token_length));
  for i := 0 to vocab_size-1 do
  begin
    FileReadWithFail(fhandle,t^.vocab_scores[i],sizeof(single));
    FileReadWithFail(fhandle,len,sizeof(len));
    t^.vocab[i] := StringOfChar(' ',len);
    FileReadWithFail(fhandle,PChar(Addr(t^.vocab[i][1]))^,len);

    if @callback <> nil then
    begin
      if not callback(vocab_size-1,i) then
        raise EFileReadError.Create('Aborted by user');
    end;
  end;
end;

procedure build_tokenizer(fhandle: integer; t: PTokenizer; vocab_size: integer; callback: TLoadCallbkFun);
begin
  t^.vocab_size := vocab_size;

  // malloc space to hold the scores and the strings
  SetLength(t^.vocab,vocab_size);
  SetLength(t^.vocab_scores,vocab_size);
  t^.sorted_vocab := nil; // initialized lazily

  // read in the file
  tokenizer_load_from_file(fhandle,t,vocab_size,callback);
end;

procedure free_tokenizer(t: PTokenizer);
begin
  t^.vocab := nil;
  t^.vocab_scores := nil;
  t^.sorted_vocab := nil;
end;

function decode(t: PTokenizer; prev_token, token: integer): string;
begin
  Result := t^.vocab[token];
  // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
  if (prev_token = 1) and (Result[1] = ' ') then
    Result := MidStr(Result,2,Length(Result));
  // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
  // parse this and convert and return the actual byte
  if LeftStr(Result,3) = '<0x' then
    Result := Chr(StrToInt('$'+MidStr(Result,4,2))); // Yeah, Delphi ;)
end;

function str_lookup(str: string; t: PTokenizer): integer;
var
  Low, Hi: Integer;
  Mid: Integer;
begin
  // efficiently find the perfect match for str in vocab, return its index or -1 if not found
  // however, Delphi doesn't have bsearch()  :'(
  Low := 0;
  Hi := t^.vocab_size-1;
  Mid := -1;
  while Low <= Hi do
  begin
    Mid := (Hi - Low) div 2 + Low;
    if t^.sorted_vocab[Mid].str = str then break;
    if t^.sorted_vocab[Mid].str < str then
      Low := Mid + 1
    else
      Hi := Mid - 1;
  end;

  if t^.sorted_vocab[Mid].str = str then
    Result := t^.sorted_vocab[Mid].id
  else
    Result := -1;
end;

function compare_tokens(a,b: pointer): integer;
var
  ia,ib: PTokenIndex;
begin
  ia := a;
  ib := b;
  if ia^.str = ib^.str then Result := 0
  else if ia^.str < ib^.str then Result := -1
  else Result := 1;
end;

procedure encode(var res: PInts; t: PTokenizer; text: string; bos, eos: boolean);
var
  i,id: integer;
  tmp: array of TTokenIndex;
  tmplst: TList;
  buf: string;
  best_score: single;
  best_id,best_idx: integer;
begin
  // encode the string text (input) into an upper-bound preallocated tokens[] array
  // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
  if text = '' then exit;

  // lazily alloc and sort the vocabulary
  if t^.sorted_vocab = nil then
  begin
    tmplst := TList.Create;
    SetLength(tmp,t^.vocab_size);
    for i := 0 to t^.vocab_size-1 do
    begin
      tmp[i].str := t^.vocab[i];
      tmp[i].id := i;
      tmplst.Add(@(tmp[i]));
    end;
    tmplst.Sort(compare_tokens);

    SetLength(t^.sorted_vocab,t^.vocab_size);
    for i := 0 to t^.vocab_size-1 do
      t^.sorted_vocab[i] := PTokenIndex(tmplst[i])^;
    tmplst.Free;
    tmp := nil;
  end;

  // add optional BOS (=1) token, if desired
  if bos then
  begin
    SetLength(res,1);
    res[High(res)] := 1;
  end;

  // add_dummy_prefix is true by default
  // so prepend a dummy prefix token to the input string, but only if text != ''
  if text <> '' then
  begin
    SetLength(res,High(res)+2);
    res[High(res)] := str_lookup(' ',t);
  end;

  // process the raw byte sequence of the input string
  for i := 1 to Length(text) do
  begin
    SetLength(res,High(res)+2);
    id := str_lookup(text[i],t);
    if id < 0 then
    begin
      // byte_fallback encoding: just encode each byte as a token
      // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
      // so the individual bytes only start at index 3
      res[High(res)] := Ord(text[i]) + 3;
    end
    else
      res[High(res)] := id;
  end;

  // merge the best consecutive pair each iteration, according the scores in vocab_scores
  while True do
  begin
    best_score := -1e10;
    best_id := -1;
    best_idx := -1;

    for i := 0 to High(res)-1 do
    begin
      // check if we can merge the pair (res[i], res[i+1])
      buf := t^.vocab[res[i]] + t^.vocab[res[i+1]];
      id := str_lookup(buf,t);
      if (id <> -1) and (t^.vocab_scores[id] > best_score) then
      begin
        // this merge pair exists in vocab! record its score and position
        best_score := t^.vocab_scores[id];
        best_id := id;
        best_idx := i;
      end;
    end;

    if best_idx = -1 then break; // we couldn't find any more pairs to merge, so we're done

    // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
    res[best_idx] := best_id;
    // delete token at position best_idx+1, shift the entire sequence back 1
    for i := best_idx+1 to High(res)-1 do
      res[i] := res[i+1];
    SetLength(res,High(res));
  end;

  // add optional EOS (=2) token, if desired
  if eos then
  begin
    SetLength(res,High(res)+2);
    res[High(res)] := 2;
  end;
end;

end.
