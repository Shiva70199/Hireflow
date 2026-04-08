from app.env import HireFlowEnv


def main() -> None:
    env = HireFlowEnv(task="hard", seed=42)
    obs = env.reset()
    print(obs.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
