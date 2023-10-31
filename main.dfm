object Form1: TForm1
  Left = 192
  Top = 118
  BorderIcons = [biSystemMenu, biMinimize]
  BorderStyle = bsSingle
  Caption = 'EMMA'
  ClientHeight = 216
  ClientWidth = 292
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
    Top = 64
    Width = 273
    Height = 113
    ScrollBars = ssVertical
    TabOrder = 2
  end
  object BitBtn1: TBitBtn
    Left = 8
    Top = 184
    Width = 113
    Height = 25
    Caption = 'Generate'
    Enabled = False
    TabOrder = 3
    OnClick = BitBtn1Click
    Kind = bkRetry
  end
  object BitBtn2: TBitBtn
    Left = 128
    Top = 184
    Width = 75
    Height = 25
    Caption = 'About'
    Enabled = False
    TabOrder = 4
    Kind = bkHelp
  end
  object BitBtn3: TBitBtn
    Left = 208
    Top = 184
    Width = 75
    Height = 25
    Caption = 'Exit'
    TabOrder = 5
    Kind = bkClose
  end
  object PBar1: TProgressBar
    Left = 8
    Top = 40
    Width = 273
    Height = 17
    TabOrder = 6
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
end
