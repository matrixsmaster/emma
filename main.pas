unit main;

interface

uses
  Windows, Messages, SysUtils, Variants, Classes, Graphics, Controls, Forms,
  Dialogs, Buttons, StdCtrls, XPMan, brain, tokenz, sampler, ComCtrls, DateUtils;

type
  TForm1 = class(TForm)
    Label1: TLabel;
    Edit1: TEdit;
    Button1: TButton;
    Memo1: TMemo;
    BitBtn1: TBitBtn;
    BitBtn2: TBitBtn;
    BitBtn3: TBitBtn;
    ODlg1: TOpenDialog;
    XPManifest1: TXPManifest;
    PBar1: TProgressBar;
    procedure Button1Click(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure FormClose(Sender: TObject; var Action: TCloseAction);
    procedure BitBtn1Click(Sender: TObject);
  private
    { Private declarations }
    floaded, aborted, loading, generating: boolean;
    trans: TTransformer;
    tokz: TTokenizer;
    sampl: TSampler;
  public
    { Public declarations }
    function ProgressCallback(total, cur: integer): boolean;
  end;

var
  Form1: TForm1;

implementation

{$R *.dfm}

function TForm1.ProgressCallback(total, cur: integer): boolean;
begin
  PBar1.Max := total;
  PBar1.Position := cur;
  Application.ProcessMessages;
  Result := not aborted;
end;

procedure TForm1.Button1Click(Sender: TObject);
var
  I: Integer;
  fhandle: integer;
  arr: PInts;
begin
  if not ODlg1.Execute then exit;
  Edit1.Text := ODlg1.FileName;

  fhandle := FileOpen(ODlg1.FileName,fmOpenRead);
  if fhandle < 0 then
  begin
    ShowMessage('Unable to open file ' + ODlg1.FileName);
    exit;
  end;

  try
    build_transformer(fhandle,@trans,ProgressCallback);
  except
    on E: Exception do
    begin
      ShowMessage('Unable to build transformer: ' + E.Message);
      exit;
    end;
  end;

  try
    build_tokenizer(fhandle,@tokz,trans.config.vocab_size,ProgressCallback);
  except
    on E: Exception do
    begin
      ShowMessage('Unable to build tokenizer: ' + E.Message);
      exit;
    end;
  end;

  FileClose(fhandle);

  build_sampler(@sampl,1.0,0.9,trans.config.vocab_size,80085);

  encode(arr,@tokz,'Testing',true,true);
  for I := 0 to High(arr) do    // Iterate
  begin
    Memo1.Lines.Add(IntToStr(arr[I]) + ' "' + tokz.vocab[arr[i]] + '"');
    Memo1.Lines.Add(FloatToStr(random_f32(sampl.rng_state)));
  end;    // for
  arr := nil;

  floaded := true;
  Button1.Enabled := false;
  BitBtn1.Enabled := true;
end;

procedure TForm1.FormCreate(Sender: TObject);
begin
  floaded := false;
  aborted := false;
  loading := false;
  generating := false;
end;

procedure TForm1.FormClose(Sender: TObject; var Action: TCloseAction);
begin
  if floaded then
  begin
    free_transformer(@trans);
    free_tokenizer(@tokz);
    free_sampler(@sampl);
  end;
end;

procedure TForm1.BitBtn1Click(Sender: TObject);
var
  i,next,token,pos,steps: integer;
  prompt_tokens: PInts;
  logits: PFloats;
  piece,output: string;
  pb: byte;
  start,stop: TDateTime;
begin
  encode(prompt_tokens,@tokz,'A girl',true,false);
  token := prompt_tokens[0]; // kick off with the first token in the prompt
  pos := 0;     // position in the sequence
  steps := 16;
  if (steps = 0) or (steps > trans.config.seq_len) then
    steps := trans.config.seq_len; // ovrerride to max length

  PBar1.Max := steps;
  PBar1.Position := 0;
  start := Time;

  while pos < steps do
  begin
    // forward the transformer to get logits for the next token
    logits := nn_ff(@trans, token, pos);

    // advance the state state machine
    if pos < High(prompt_tokens) then
    begin
      // if we are still processing the input prompt, force the next prompt token
      next := prompt_tokens[pos + 1];
    end
    else
    begin
      // otherwise sample the next token from the logits
      next := sample(@sampl, logits);
    end;
    inc(pos);

    // data-dependent terminating condition: the BOS (=1) token delimits sequences
    if next = 1 then break;

    // print token as string, decode it with the Tokenizer object: integer
    piece := decode(@tokz, token, next);
    for i := 1 to Length(piece) do
    begin
      pb := Ord(piece[i]);
      if (pb > 31) and (pb < 127) then
        output := output + piece[i]
      else if pb = 10 then
        output := output + '\r\n'
      else
        output := output + '<+' + IntToStr(pb) + '>';
      token := next;
    end;

    PBar1.Position := pos;
    Application.ProcessMessages;
  end;

  stop := Time;
  Memo1.Lines.Add(output);
  Memo1.Lines.Add(FloatToStr(MilliSecondSpan(stop,start)/steps) + ' ms/token');
end;

end.
