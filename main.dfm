object Form1: TForm1
  Left = 192
  Top = 118
  BorderIcons = [biSystemMenu, biMinimize]
  BorderStyle = bsSingle
  Caption = 'EMMA'
  ClientHeight = 241
  ClientWidth = 289
  Color = clBtnFace
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -11
  Font.Name = 'MS Sans Serif'
  Font.Style = []
  OldCreateOrder = False
  Position = poDesktopCenter
  OnClose = FormClose
  OnCreate = FormCreate
  PixelsPerInch = 96
  TextHeight = 13
  object Label1: TLabel
    Left = 8
    Top = 8
    Width = 38
    Height = 13
    Caption = 'VNA file'
  end
  object Label2: TLabel
    Left = 8
    Top = 64
    Width = 33
    Height = 13
    Caption = 'Prompt'
  end
  object Label3: TLabel
    Left = 8
    Top = 88
    Width = 33
    Height = 13
    Caption = 'Length'
  end
  object Label4: TLabel
    Left = 256
    Top = 88
    Width = 6
    Height = 13
    Caption = '8'
  end
  object Edit1: TEdit
    Left = 56
    Top = 8
    Width = 193
    Height = 21
    ReadOnly = True
    TabOrder = 0
  end
  object Button1: TButton
    Left = 256
    Top = 8
    Width = 25
    Height = 25
    Caption = '...'
    TabOrder = 1
    OnClick = Button1Click
  end
  object Memo1: TMemo
    Left = 8
    Top = 112
    Width = 273
    Height = 89
    ScrollBars = ssVertical
    TabOrder = 5
  end
  object BitBtn1: TBitBtn
    Left = 8
    Top = 208
    Width = 113
    Height = 25
    Caption = 'Generate'
    Enabled = False
    TabOrder = 6
    OnClick = BitBtn1Click
    Kind = bkRetry
  end
  object BitBtn2: TBitBtn
    Left = 128
    Top = 208
    Width = 75
    Height = 25
    Caption = 'About'
    TabOrder = 7
    OnClick = BitBtn2Click
    Kind = bkHelp
  end
  object BitBtn3: TBitBtn
    Left = 208
    Top = 208
    Width = 75
    Height = 25
    Caption = 'Exit'
    TabOrder = 8
    Kind = bkClose
  end
  object PBar1: TProgressBar
    Left = 8
    Top = 40
    Width = 273
    Height = 17
    TabOrder = 2
  end
  object Edit2: TEdit
    Left = 48
    Top = 64
    Width = 233
    Height = 21
    TabOrder = 3
    Text = 'Once upon a time'
  end
  object TrackBar1: TTrackBar
    Left = 48
    Top = 88
    Width = 201
    Height = 17
    Max = 256
    Min = 8
    Position = 8
    TabOrder = 4
    ThumbLength = 12
    OnChange = TrackBar1Change
  end
  object ODlg1: TOpenDialog
    DefaultExt = 'vna'
    Filter = 'Virtual Diana Files|*.vna|All files|*.*'
    Title = 'Open VNA'
    Left = 136
    Top = 8
  end
  object XPManifest1: TXPManifest
    Left = 96
    Top = 8
  end
  object Timer1: TTimer
    Interval = 100
    OnTimer = Timer1Timer
    Left = 176
    Top = 8
  end
end
