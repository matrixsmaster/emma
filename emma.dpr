program emma;

uses
  Forms,
  main in 'main.pas' {Form1},
  brain in 'brain.pas',
  tokenz in 'tokenz.pas',
  sampler in 'sampler.pas';

{$R *.res}

begin
  Application.Initialize;
  Application.CreateForm(TForm1, Form1);
  Application.Run;
end.
