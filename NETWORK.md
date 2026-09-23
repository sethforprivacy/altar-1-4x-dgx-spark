# Four-Spark network preparation

This recipe uses SparkRing's patched NCCL over a physical `0–1–2–3–0` RoCE
ring. Keep a separate management Ethernet connection for SSH, rendezvous and
recovery. Rank order in `site.json` must follow the cable order.

Use SparkRing commit `f16b5f43208c188c018c43820ccf52955006d1a5` for the
host-preparation tools reviewed with this recipe:

```sh
git clone https://github.com/FujitsuPolycom/sparkring.git
cd sparkring
git checkout --detach f16b5f43208c188c018c43820ccf52955006d1a5
less bootstrap.sh
bash bootstrap.sh --ref f16b5f43208c188c018c43820ccf52955006d1a5
export PATH="$HOME/.local/bin:$PATH"
sparkring host check
sparkring cluster init --size 4
sparkring cluster configure
```

The final command prints the proposed fabric configuration. Check that it names
only the intended data interfaces and that its subnet allocation does not
overlap the management network or VPN. For a new, idle fabric, apply the
reviewed configuration and inspect routing:

```sh
sparkring cluster configure --apply
sparkring doctor --verify
sparkring doctor --verify --apply
sparkring doctor --verify
```

These commands configure hosts, not Altar. Read the complete pinned
[bootstrap procedure](https://github.com/FujitsuPolycom/sparkring/blob/f16b5f43208c188c018c43820ccf52955006d1a5/docs/operations/bootstrap.md)
for enrollment, host-key verification, repair boundaries and persistence.
Existing working fabrics should be inspected rather than readdressed.

The tested configuration exposes four RDMA functions per Spark:

```text
rocep1s0f0:1
rocep1s0f1:1
roceP2p1s0f0:1
roceP2p1s0f1:1
```

Names, connected ports, addresses and GID indices are host-specific. Confirm
actual mappings with `rdma link show`, `ibdev2netdev`, `ip -br address` and
`show_gids`. The tested GID index is 3. Do not assume that index remains correct
after firmware or network changes. The shared bootstrap configures primary
interfaces; the pinned [four-function host setup](https://github.com/FujitsuPolycom/sparkring/blob/f16b5f43208c188c018c43820ccf52955006d1a5/docs/GLM53_SPARK_MESH_HOST_SETUP.md)
covers secondary functions and host configuration. Its model-launch commands
are for a different GLM profile; launch Altar with this repository's scripts.

Set `NCCL_IB_HCA` to the verified, exact port list with a leading `=`. Set
`SOCKET_IFNAME` to management Ethernet. `run-rank.sh` selects IB, subnet-aware
routing and ring collectives using the pinned image's bundled NCCL. Its
`ag_rs` decode-context backend uses all-gather/reduce-scatter; the default
all-to-all path tried non-neighbor RDMA links on the tested fabric and failed.

Before loading weights, run the `ring-probe` mode on all four ranks and require
its exact-value all-reduce, all-gather and reduce-scatter checks to pass on
every rank. Inspect NCCL logs for `Using network IB` and the intended ports.
This separates collective correctness from model startup and API tests.
