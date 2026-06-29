#define MyAppName "VideoLingo Q"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "qxk2005"
#define MyAppURL "https://github.com/qxk2005/videolingo-q"
#define MyAppExeName "start_windows.bat"

[Setup]
; 注意: AppId 用于唯一标识该应用，请勿更改。
AppId={{D377B8A0-E3B5-4D56-8A67-1B9E8F3C5A20}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DisableProgramGroupPage=yes
; 关闭某些不需要的向导页，以简化安装流程
DisableReadyPage=no
DisableFinishedPage=no
; 输出路径和文件名
OutputDir=output
OutputBaseFilename=VideoLingo-Q-Setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "chinesesimp"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; 将整个打包好的目录拷贝进安装包
Source: "VideoLingo-Q-Windows-x64\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{sys}\cmd.exe"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{sys}\cmd.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: shellexec postinstall skipifsilent
