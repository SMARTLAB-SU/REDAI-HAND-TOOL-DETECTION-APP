; Inno Setup Script for REDAI Hand Tool Detection App
[Setup]
AppName=REDAI Hand Tool Detection App
AppVersion=1.0
DefaultDirName={autopf}\REDAI Hand Tool Detection App
DefaultGroupName=REDAI Hand Tool Detection App
UninstallDisplayIcon={app}\App.exe
Compression=lzma
SolidCompression=yes

[Files]
Source: "..\App.exe"; DestDir: "{app}"
Source: "..\Source Code\*"; DestDir: "{app}\Source Code"; Flags: recursesubdirs

[Icons]
Name: "{autoprogram}\REDAI Hand Tool Detection App"; Filename: "{app}\App.exe"
