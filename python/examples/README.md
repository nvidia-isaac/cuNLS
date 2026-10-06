# pycunls examples

End-to-end programs using `pycunls`. Each script keeps its `pycunls` calls inline;
`example_utils/` holds everything else (synthetic data, metrics, SE(3) helpers,
visualization).

| Example | What it shows |
|---|---|
| `sparse_bundle_adjustment.py` | Joint camera-pose and landmark optimization with CuPy |
| `pose_graph_optimization.py` | SE(3) pose-graph optimization with CuPy |
| `custom_warp_factor.py` | A custom factor kernel written with NVIDIA Warp |
| `custom_warp_state.py` | A custom state batch (positive-scalar manifold) written with NVIDIA Warp |
| `ransac_pnp.py` | Robust PnP with 50% outliers using `RansacLevenbergMarquardtMinimizer` |
| `imu_bundle_adjustment.py` | Visual-inertial bundle adjustment with `ImuFactorBatch` |
| [`tartan_vio.py`](#visual-inertial-odometry-with-ransac-on-tartanground) | RGB-D inertial odometry on TartanGround with injected outliers: IMU factor + RANSAC |
| [`supermarket_drones.py`](#a-drone-fleet-in-a-supermarket-batched-mpc) | A fleet of quadrotors in a real store with walking shoppers: batched MPC |

The two showcases below need real data from [TartanGround](https://tartanair.org/tartanground/)
and log to [Rerun](https://rerun.io) (`pip install rerun-sdk opencv-python scipy`).

## Visual-inertial odometry with RANSAC on TartanGround

![tartan_vio](assets/tartan_vio.gif)

A legged robot walks 82 m through `OldTownFall` (sequence `Data_anymal/P2000`). Its front
camera (640 x 640, 10 Hz, with depth) and IMU (100 Hz) drive a small odometry:

- **front end** (OpenCV): KLT feature tracks; each track's landmark comes from the depth
  image at its first frame and stays fixed (a frame-to-map odometry);
- **back end** (cuNLS): every frame, one 30-dimensional problem on a two-frame window: pose,
  velocity and IMU bias of the previous and the current frame. `ImuFactorBatch` and the priors
  from the last solve are *always on*; one `PnPFactorBatch` factor per tracked landmark is
  *sampled* and classified by `RansacLevenbergMarquardtMinimizer`. The IMU predicts the motion,
  so two matches per hypothesis are enough, and mismatches that agree with a motion the IMU
  rules out are rejected.

Outliers are injected into the 2D matches the back end sees: swapped matches, a coherent
shift of a third of the tracks (repetitive texture), a camera blackout and 75% clutter.
Noise and biases are added to the dataset's ideal IMU. Visual-only RANSAC, the same window
with a Huber loss instead of RANSAC, and IMU dead reckoning run on the same data:

| method | final position error (82 m walked) |
|---|---|
| **inertial RANSAC (cuNLS)** | **0.72 m (0.9%)**, 5 of 58,106 injected outliers accepted |
| visual-only RANSAC | 2.36 m (2.9%) |
| inertial LM + Huber loss | diverges in the clutter segment |
| IMU dead reckoning | lost within seconds |

```bash
# Data (pip install tartanair; about 1.1 GB)
python -c "import tartanair as ta; ta.init('dataset/tartan_ground'); \
  ta.download_ground(env=['OldTownFall'], version=['anymal'], traj=['P2000'], \
  modality=['image', 'depth', 'imu'], camera_name=['lcam_front'], unzip=True)"

python tartan_vio.py --data dataset/tartan_ground/OldTownFall/Data_anymal/P2000 --spawn
# or --rrd tartan_vio.rrd, then: rerun tartan_vio.rrd
```

## A drone fleet in a supermarket: batched MPC

![supermarket_drones](assets/supermarket_drones.gif)

The TartanGround `Supermarket` is fused from the depth images of its ten trajectories into a
3D map (`example_utils/tartan_map.py`). Shoppers walk the paths the dataset's robot drove,
and a fleet of quadrotors flies deliveries between stations along A* routes through the
aisles. The whole fleet is **one batched MPC problem** (`pycunls.mpc.Horizon(batch=B)`, 40
steps of 25 ms, `QuadrotorFactorBatch` dynamics, thrust bounds), solved together every
control step with each drone its own subproblem. On every step of its horizon each drone
keeps clear of

- the map voxels nearest to its previous plan (shelves, pillars, the floor),
- the shoppers, grown by a 0.6 m personal space and predicted at constant velocity,
- the other drones' previous plans, with a 0.5 m separation zone,

all as sphere clearance constraints rewritten in `Horizon.obstacles` every step. The left
panel shows the store from above, the right one follows drone 0 from behind.

With 6 drones and 10 shoppers over 45 s (RTX A6000): 15 deliveries, drones at least 1.06 m
apart (1.15 kept), at least 0.90 m from a shopper's body (1.00 kept; the gap is prediction
error), about 30 ms per fleet solve. The GIF shows 10 drones and 20 shoppers.

```bash
# Data (about 8 GB: the image and depth of every Supermarket trajectory)
python -c "import tartanair as ta; ta.init('dataset/tartan_ground'); \
  ta.download_ground(env=['Supermarket'], version=['omni', 'diff'], \
  modality=['image', 'depth'], camera_name=['lcam_front'], unzip=True)"

python supermarket_drones.py --env dataset/tartan_ground/Supermarket --spawn \
    --drones 10 --shoppers 20 --duration 30
```

The first run fuses the map (about a minute) and caches it as `fused_map.npz` in the
environment folder (`--map` to choose the file).
