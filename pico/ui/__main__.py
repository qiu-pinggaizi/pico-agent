"""Allow: python -m pico.ui.training_server [--port N] [--scan DIR ...]"""
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Pico Training Monitor")
    parser.add_argument("--port", "-p", type=int, default=8766)
    parser.add_argument("--scan", nargs="*", help="Directories to scan")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    from pico.ui.training_server import start_monitor

    start_monitor(
        host=args.host,
        port=args.port,
        scan_roots=args.scan or None,
        open_browser=not args.no_open,
    )


if __name__ == "__main__":
    main()
