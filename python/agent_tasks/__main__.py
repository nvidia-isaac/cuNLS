# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Command line of the agent tasks.

    python -m agent_tasks list
    python -m agent_tasks prepare TASK DIR [--seed N]     # writes DIR/data.npz and DIR/PROMPT.md
    python -m agent_tasks check TASK ANSWER.json [--seed N]
    python -m agent_tasks reference TASK DIR [--seed N]   # writes DIR/answer.json (the reference)
"""

import argparse
import json
import pathlib
import sys

import numpy as np

from agent_tasks.tasks import TASKS


def _prompt(task, directory: pathlib.Path) -> str:
    return (f"{task.prompt}\n\nThe data is in {directory / 'data.npz'} (numpy.load). Use the "
            "pycunls library (GPU nonlinear least squares). Write the answer as JSON to "
            f"{directory / 'answer.json'}.\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent_tasks")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    for name in ("prepare", "reference"):
        p = sub.add_parser(name)
        p.add_argument("task", choices=sorted(TASKS))
        p.add_argument("directory", type=pathlib.Path)
        p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("check")
    p.add_argument("task", choices=sorted(TASKS))
    p.add_argument("answer", type=pathlib.Path)
    p.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.command == "list":
        for name, task in TASKS.items():
            print(f"{name}: {task.prompt}")
        return 0
    task = TASKS[args.task]
    data, truth = task.make(args.seed)
    if args.command == "prepare":
        args.directory.mkdir(parents=True, exist_ok=True)
        np.savez(args.directory / "data.npz", **data)
        (args.directory / "PROMPT.md").write_text(_prompt(task, args.directory.resolve()))
        print(_prompt(task, args.directory.resolve()))
        return 0
    if args.command == "reference":
        args.directory.mkdir(parents=True, exist_ok=True)
        loaded = dict(np.load(args.directory / "data.npz")) if (
            args.directory / "data.npz").exists() else data
        (args.directory / "answer.json").write_text(json.dumps(task.reference(loaded)))
        return 0
    try:
        answer = json.loads(args.answer.read_text())
    except (OSError, ValueError) as exc:
        print(json.dumps({"task": args.task, "seed": args.seed, "passed": False,
                          "message": f"cannot read the answer: {exc}"}))
        return 1
    passed, message = task.check(answer, truth)
    print(json.dumps({"task": args.task, "seed": args.seed, "passed": passed,
                      "message": message}))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
