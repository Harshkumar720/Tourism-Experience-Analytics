import os
import sys
from pathlib import Path


def main():
    # Project root directory
    project_root = Path(__file__).resolve().parent

    # Path to Streamlit app
    streamlit_app = project_root / "app" / "streamlit_app.py"

    # Check whether the Streamlit app exists
    if not streamlit_app.exists():
        print("Error: app/streamlit_app.py not found.")
        print(f"Expected location: {streamlit_app}")
        sys.exit(1)

    print("=" * 60)
    print("Tourism Experience Analytics")
    print("Launching Streamlit application...")
    print("=" * 60)

    # Launch Streamlit
    os.system(f'streamlit run "{streamlit_app}"')


if __name__ == "__main__":
    main()