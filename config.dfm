object Form2: TForm2
  Left = 306
  Top = 120
  BorderIcons = [biSystemMenu]
  BorderStyle = bsToolWindow
  Caption = 'Configuration'
  ClientHeight = 161
  ClientWidth = 289
  Color = clBtnFace
  Font.Charset = DEFAULT_CHARSET
  Font.Color = clWindowText
  Font.Height = -11
  Font.Name = 'MS Sans Serif'
  Font.Style = []
  OldCreateOrder = False
  Position = poMainFormCenter
  OnCreate = FormCreate
  PixelsPerInch = 96
  TextHeight = 13
  object Label1: TLabel
    Left = 8
    Top = 8
    Width = 27
    Height = 13
    Caption = 'Temp'
  end
  object Label2: TLabel
    Left = 8
    Top = 48
    Width = 29
    Height = 13
    Caption = 'Top P'
  end
  object Label3: TLabel
    Left = 240
    Top = 8
    Width = 6
    Height = 13
    Caption = '0'
  end
  object Label4: TLabel
    Left = 240
    Top = 48
    Width = 6
    Height = 13
    Caption = '0'
  end
  object Label5: TLabel
    Left = 8
    Top = 88
    Width = 25
    Height = 13
    Caption = 'Seed'
  end
  object TrackBar1: TTrackBar
    Left = 48
    Top = 8
    Width = 185
    Height = 25
    Max = 100
    TabOrder = 0
    ThumbLength = 16
    OnChange = TrackBar1Change
  end
  object TrackBar2: TTrackBar
    Left = 48
    Top = 48
    Width = 185
    Height = 25
    Max = 100
    TabOrder = 1
    ThumbLength = 16
    OnChange = TrackBar2Change
  end
  object SpinEdit1: TSpinEdit
    Left = 56
    Top = 88
    Width = 121
    Height = 22
    MaxValue = 0
    MinValue = 0
    TabOrder = 2
    Value = 0
  end
  object BitBtn1: TBitBtn
    Left = 8
    Top = 128
    Width = 273
    Height = 25
    Caption = 'Apply'
    TabOrder = 3
    OnClick = BitBtn1Click
    Kind = bkOK
  end
end
