from repo2env.openenv.server import app, run_server


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the Repo2Env OpenEnv server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()


__all__ = ["app", "main"]
