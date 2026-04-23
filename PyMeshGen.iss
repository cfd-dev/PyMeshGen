; ============================================================================
; PyMeshGen Inno Setup 安装脚本
; ============================================================================
; 使用方法:
;   1. 先运行 build.bat 构建可执行文件
;   2. 安装 Inno Setup: https://jrsoftware.org/isdl.php
;   3. 运行: iscc.exe PyMeshGen.iss
; ============================================================================

#define MyAppName "PyMeshGen"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "PyMeshGen Team"
#define MyAppExeName "PyMeshGen.exe"
#define MyAppURL "https://github.com/pymeshgen/PyMeshGen"

[Setup]
; 基本设置
AppId={{A3B5E1C2-4D6F-8A9B-0C1D-2E3F4A5B6C7D}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
LicenseFile=LICENSE.txt
OutputDir=installer
OutputBaseFilename=PyMeshGen-Setup-{#MyAppVersion}
SetupIconFile=docs\icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

; 语言设置
Languages:
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"
Name: "english"; MessagesFile: "compiler:Languages\English.isl"

; 安装步骤
[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 6.1; Check: not IsAdminInstallMode

[Files]
; 主程序文件
Source: "dist\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; 注意：不要复制以下目录，它们已经包含在 dist 中
; Source: "config\*"; DestDir: "{app}\config"; Flags: ignoreversion recursesubdirs createallsubdirs
; Source: "docs\*"; DestDir: "{app}\docs"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: quicklaunchicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
function InitializeSetup(): Boolean;
var
  ResultCode: Integer;
begin
  Result := True;
  
  // 检查是否已安装
  if RegKeyExists(HKLM, 'SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\{#SetupSetting("AppId")}_is1') then
  begin
    if MsgBox('PyMeshGen 已经安装。是否要卸载旧版本？', mbConfirmation, MB_YESNO) = IDYES then
    begin
      // 运行卸载程序
      ShellExec('open', ExpandConstant('{uninstallexe}'), '', '', SW_SHOWNORMAL, ewWaitUntilTerminated, ResultCode);
    end
    else
    begin
      Result := False;
    end;
  end;
end;

function IsVCRedistInstalled(): Boolean;
begin
  // 检查 Visual C++ Redistributable 是否已安装
  Result := RegKeyExists(HKLM, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64');
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    // 如果未安装 VC++ Redistributable，提示用户
    if not IsVCRedistInstalled() then
    begin
      if MsgBox('检测到系统未安装 Visual C++ Redistributable，是否需要下载？' + #13#10 + 
                'PyMeshGen 需要此组件才能运行。', mbConfirmation, MB_YESNO) = IDYES then
      begin
        ShellExec('open', 'https://aka.ms/vs/17/release/vc_redist.x64.exe', '', '', SW_SHOWNORMAL, ewNoWait, ResultCode);
      end;
    end;
  end;
end;
