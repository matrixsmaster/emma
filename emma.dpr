program emma;

uses
  Forms,
  main in 'main.pas' {Form1},
  brain in 'brain.pas',
  tokenz in 'tokenz.pas',
  sampler in 'sampler.pas',
  config in 'config.pas' {Form2};

{$R *.res}

begin
  Application.Initialize;
  Application.Title := 'EMMA';
  Application.CreateForm(TForm1, Form1);
  Application.CreateForm(TForm2, Form2);
  Application.Run;
end.
