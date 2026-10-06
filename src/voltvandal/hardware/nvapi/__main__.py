"""CLI shim: python -m voltvandal.hardware.nvapi -curve <gpu> <op> [file]"""

import sys

from .curves import apply_curve, dump_curve
from .native import _get_handle, _nvapi_init, _reset_curve


#   python -m voltvandal.hardware.nvapi -curve <gpu> -1 <out.csv>   (dump)
#   python -m voltvandal.hardware.nvapi -curve <gpu>  1 <in.csv>    (apply)
#   python -m voltvandal.hardware.nvapi -curve <gpu>  0             (reset to defaults)
def main() -> None:
    args = sys.argv[1:]
    if len(args) >= 3 and args[0].lower() == "-curve":
        try:
            gpu = int(args[1])
            op  = int(args[2])
        except ValueError:
            print("Usage: nvapi -curve <gpu> <op> [file]",
                  file=sys.stderr)
            sys.exit(1)

        if op == -1:
            if len(args) < 4:
                print("-curve <gpu> -1 requires a filename", file=sys.stderr)
                sys.exit(1)
            dump_curve(gpu, args[3])
        elif op == 1:
            if len(args) < 4:
                print("-curve <gpu> 1 requires a filename", file=sys.stderr)
                sys.exit(1)
            apply_curve(gpu, args[3])
        elif op == 0:
            _nvapi_init()
            _reset_curve(_get_handle(gpu))
        else:
            print(f"Unknown operation: {op}", file=sys.stderr)
            sys.exit(1)
    else:
        print(
            "Usage:\n"
            "  python -m voltvandal.hardware.nvapi -curve <gpu> -1 <out.csv>   (dump VF curve)\n"
            "  python -m voltvandal.hardware.nvapi -curve <gpu>  1 <in.csv>    (apply VF curve)\n"
            "  python -m voltvandal.hardware.nvapi -curve <gpu>  0             (reset to defaults)",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
