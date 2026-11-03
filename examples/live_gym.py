"""Watch a Gymnasium policy think, live: stream every tick to the netglow viewer while the env runs.

    pip install "git+https://github.com/ahuou/netglow#subdirectory=python" "stable-baselines3" "gymnasium[box2d]"
    python examples/live_gym.py                                  # trains a small CartPole PPO first (~20 s)
    python examples/live_gym.py --sb3 ppo.zip --env LunarLander-v3

Then open the link it prints (the hosted page with ?live=8765, or http://localhost:8765/ served from here).
The browser runs the model itself on each streamed observation; press Record there to save what you streamed.
"""

import argparse

import netglow


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sb3", help="a Stable-Baselines3 .zip (PPO, A2C, DQN, SAC, TD3); default: train a quick PPO")
    ap.add_argument("--env", default="CartPole-v1")
    ap.add_argument("--steps", type=int, default=5000, help="ticks to stream (episodes restart)")
    ap.add_argument("--train-steps", type=int, default=30000)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--out", default=None, help="also save model.onnx + recording.npz + labels.json here")
    a = ap.parse_args()

    model = a.sb3
    if model is None:
        from stable_baselines3 import PPO

        print(f"[demo] training PPO on {a.env} for {a.train_steps} steps")
        model = PPO("MlpPolicy", a.env, seed=0, device="cpu").learn(a.train_steps)

    # live=True: start the server, pace the env at its own fps, stream each tick; frames are the env's render
    netglow.record(model, a.env, steps=a.steps, live=True, port=a.port, out_dir=a.out)


if __name__ == "__main__":
    main()
