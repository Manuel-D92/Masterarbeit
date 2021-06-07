# -*- mode: python -*-

block_cipher = pyi_crypto.PyiBlockCipher(key='xxxxxxxxxxxxxxxx')

import sys
sys.setrecursionlimit(10000)

a = Analysis(['main.py'],
             pathex=['C:\\Users\\muellead\\Documents\\hockey_glt'],
             binaries=None,
             datas=None,
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=['config'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='main',
          debug=False,
          strip=False,
          upx=True,
          console=False )
