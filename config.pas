unit config;

interface

uses
  Windows, Messages, SysUtils, Variants, Classes, Graphics, Controls, Forms,
  Dialogs, StdCtrls, Buttons, Spin, ComCtrls;

type
  TForm2 = class(TForm)
    Label1: TLabel;
    Label2: TLabel;
    TrackBar1: TTrackBar;
    TrackBar2: TTrackBar;
    Label3: TLabel;
    Label4: TLabel;
    Label5: TLabel;
    SpinEdit1: TSpinEdit;
    BitBtn1: TBitBtn;
    BitBtn2: TBitBtn;
    procedure TrackBar1Change(Sender: TObject);
    procedure BitBtn1Click(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure TrackBar2Change(Sender: TObject);
    procedure BitBtn2Click(Sender: TObject);
  private
    { Private declarations }
  public
    { Public declarations }
    temp: real;
    topp: real;
  end;

const
  c_temp = 30;
  c_topp = 90;

var
  Form2: TForm2;

implementation
uses main;

{$R *.dfm}

procedure TForm2.TrackBar1Change(Sender: TObject);
begin
  temp := TrackBar1.Position / 100.0;
  if temp = 0 then Label3.Caption := 'Greedy'
  else Label3.Caption := FloatToStr(temp);
end;

procedure TForm2.BitBtn1Click(Sender: TObject);
begin
  Form1.UpdateSamplerConfig;
end;

procedure TForm2.FormCreate(Sender: TObject);
begin
  TrackBar1.Position := c_temp;
  TrackBar2.Position := c_topp;
  TrackBar1Change(Sender);
  TrackBar2Change(Sender);
end;

procedure TForm2.TrackBar2Change(Sender: TObject);
begin
  topp := TrackBar2.Position / 100.0;
  Label4.Caption := FloatToStr(topp);
end;

procedure TForm2.BitBtn2Click(Sender: TObject);
begin
  SpinEdit1.Value := random(99999);
end;

end.
