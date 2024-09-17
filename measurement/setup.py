"""Install this Python package."""

from pathlib import Path

from setuptools import setup


class Setup:
    """Convenience wrapper (for C.I. purposes) of the `setup()` call form `setuptools`.

    It automatically creates executables for package from .py files (except __init__.py).
    """

    def __init__(self):
        """Initialize attributes for entrypoints discovery."""
        self.conf = {}
        # Automatically produce executables based on the content of the exe module
        self.conf["entry_points"] = {"console_scripts": []}
        for exe in Path.cwd().glob("src/**/exe/*.py"):
            exe = exe.relative_to(Path.cwd())
            if exe.name == "__init__.py":
                continue

            final_exe = (
                "-".join(x.replace("_", "-") for x in exe.parent.parent.parts[1:]) + "-" + exe.stem.replace("_", "-")
            )
            module = ".".join(exe.parent.parts[1:]) + "." + exe.stem
            self.conf["entry_points"]["console_scripts"].append(f"{final_exe} = {module}:main")

    def __str__(self):
        """Return a stringified version of the current configuration."""
        return str(self.conf)

    def __call__(self):
        """Run the setup when calling the class instance as if it were a function."""
        setup(**self.conf)


SETUP = Setup()

if __name__ == "__main__":
    SETUP()
