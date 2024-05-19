unit main;

interface

uses
  Windows, Messages, SysUtils, Variants, Classes, Graphics, Controls, Forms,
  Dialogs, Buttons, StdCtrls, XPMan, brain, tokenz, sampler, ComCtrls, DateUtils,
  ExtCtrls;

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
    Timer1: TTimer;
    Edit2: TEdit;
    Label2: TLabel;
    Label3: TLabel;
    TrackBar1: TTrackBar;
    Label4: TLabel;
    procedure Button1Click(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure FormClose(Sender: TObject; var Action: TCloseAction);
    procedure BitBtn1Click(Sender: TObject);
    procedure Timer1Timer(Sender: TObject);
    procedure BitBtn2Click(Sender: TObject);
    procedure TrackBar1Change(Sender: TObject);
  private
    { Private declarations }
    floaded, aborted, loading, generating: boolean;
    trans: TTransformer;
    tokz: TTokenizer;
    sampl: TSampler;
    seed_val: integer;
  public
    { Public declarations }
    function ProgressCallback(total, cur: integer): boolean;
    procedure UpdateSamplerConfig;
  end;

var
  Form1: TForm1;

implementation
uses config;

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

  floaded := true;
  Button1.Enabled := false;
  BitBtn1.Enabled := true;
  Form2.Show;
  Form2.SpinEdit1.Value := seed_val;
  Timer1.Enabled := false;
  UpdateSamplerConfig;
end;

procedure TForm1.FormCreate(Sender: TObject);
begin
  floaded := false;
  aborted := false;
  loading := false;
  generating := false;
  seed_val := 0;
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
  if Edit2.Text <> '' then
    encode(prompt_tokens,@tokz,Edit2.Text,true,false)
  else
    encode(prompt_tokens,@tokz,'A girl',true,false);
  token := prompt_tokens[0]; // kick off with the first token in the prompt
  pos := 0;     // position in the sequence
  steps := TrackBar1.Position;
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
      // if we are still processing the input prompt, force the next prompt token
      next := prompt_tokens[pos + 1]
    else
      // otherwise sample the next token from the logits
      next := sample(@sampl, logits);
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
      else
      if (pb = 10) or (pb = 13) then
      begin
        if output <> '' then Memo1.Lines.Add(output);
        output := '';
      end
      else
        output := output + '<+' + IntToStr(pb) + '>';
      token := next;
    end;

    PBar1.Position := pos;
    Application.ProcessMessages; // this will mess with time measurement, but oh well...
  end;

  stop := Time;
  if output <> '' then Memo1.Lines.Add(output);
  Memo1.Lines.Add('Speed: ' + FloatToStr(MilliSecondSpan(stop,start)/steps) + ' ms/token');
end;

procedure TForm1.Timer1Timer(Sender: TObject);
begin
  Inc(seed_val);
end;

procedure TForm1.BitBtn2Click(Sender: TObject);
begin
  ShowMessage('EMMA is AI inference engine written in Delphi 7 for Windows XP'+#13+
              '(C) Dmitry ''sciloaf'' Solovyev, 2023-2024'+#13+#13+
              'Based on ''llama2.c'' (C) Andrej Karpathy, 2023');
end;

procedure TForm1.TrackBar1Change(Sender: TObject);
begin
  Label4.Caption := IntToStr(TrackBar1.Position);
end;

procedure TForm1.UpdateSamplerConfig;
begin
  free_sampler(@sampl);
  build_sampler(@sampl,Form2.temp,Form2.topp,trans.config.vocab_size,Form2.SpinEdit1.Value);
  Memo1.Lines.Add('Seed value = ' + IntToStr(Form2.SpinEdit1.Value));
end;

end.
