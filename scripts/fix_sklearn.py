"""Fix sklearn version — jalankan dengan: python scripts/fix_sklearn.py"""

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    python = sys.executable
    print(f"Python executable: {python}")

    print("Installing numpy 1.26.4...")
    step1 = subprocess.run(
        [python, "-m", "pip", "install", "numpy==1.26.4"],
        check=False,
    )
    if step1.returncode != 0:
        print("GAGAL install numpy — coba manual:")
        print(f'  "{python}" -m pip install numpy==1.26.4')
        return

    print("Installing scikit-learn 1.3.2...")
    result = subprocess.run(
        [python, "-m", "pip", "install", "scikit-learn==1.3.2"],
        check=False,
    )

    if result.returncode == 0:
        import sklearn

        print(f"SUCCESS: scikit-learn {sklearn.__version__} terinstall")
    else:
        print("GAGAL — coba manual:")
        print(f'  "{python}" -m pip install scikit-learn==1.3.2')


if __name__ == "__main__":
    main()
