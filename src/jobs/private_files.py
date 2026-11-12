"""Provision the managed root for the current operator and SYSTEM only on Windows."""

import os
import subprocess
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=128)
def secure_root(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "nt":
        script = """
$ErrorActionPreference = 'Stop'
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
$acl = New-Object System.Security.AccessControl.DirectorySecurity
$acl.SetOwner($identity)
$acl.SetAccessRuleProtection($true, $false)
foreach ($sid in @($identity, [System.Security.Principal.SecurityIdentifier]'S-1-5-18')) {
  $rule = New-Object System.Security.AccessControl.FileSystemAccessRule(
    $sid, 'FullControl', 'ContainerInherit,ObjectInherit', 'None', 'Allow')
  $acl.AddAccessRule($rule)
}
[System.IO.Directory]::SetAccessControl($env:VA_LEARNING_PRIVATE_ROOT, $acl)
"""
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            env={**os.environ, "VA_LEARNING_PRIVATE_ROOT": str(root)},
            check=True,
            capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    else:
        root.chmod(0o700)
