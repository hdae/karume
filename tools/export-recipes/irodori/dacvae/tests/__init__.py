"""DACVAE recipe のテスト。

パッケージにしてあるのは、family が増えたときに `<family>/tests/test_export.py` の
**basename が衝突しない**ようにするため（pytest の既定 import mode は `__init__.py` の
無いディレクトリのテストを basename だけのモジュール名で読む）。ここは
`irodori/tests/test_export.py` と 1 段ずれた同名なので、なおさら要る。
"""
