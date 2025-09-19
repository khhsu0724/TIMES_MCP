from mcp.server.fastmcp import Context, FastMCP
from mcp.types import TextContent, ImageContent
from typing import Literal, Dict, Any, Optional
import logging, os, asyncio, tempfile, hashlib, time
import io, base64
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any
from inputs import *
from plot import *
from util import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP()

# @mcp.tool(
#     name = "list_env",
#     description = "List Environmental variables"
# )
# def list_env() -> dict:
#     """Return all environment variables visible to this MCP server."""
#     return dict(os.environ)

def _make_run_dir(base: Optional[str] = None) -> Path:
    """Create a unique run directory under base (or system temp)."""
    if base is None:
        base = tempfile.gettempdir()
    base = Path(base)

    uniq = hashlib.sha1(f"{time.time_ns()}_{os.getpid()}".encode()).hexdigest()[:12]
    run_dir = base / "mcpruns" / uniq   # <-- no leading slash
    run_dir.mkdir(parents=True, exist_ok=False)
    return Path(run_dir)

@mcp.tool(
    name="get_multiplet_ground_state", 
    description="return ground state information of a multiplet calculation"
)

def get_multiplet_ground_state(
    run_dir: str,
    ) -> Dict[str, Any]:
    """
    Extracts the multiplet ground state information from a calculation output directory.

    This function parses the output files in the given run directory and returns a dictionary
    containing both the electronic occupation numbers and the ground state composition.

    Parameters
    ----------
    run_dir : str
        Path to the calculation output directory containing the relevant files.

    Returns
    -------
    Dict[str, Any]
        {
            "Occupation": dict    # orbital occupation of the ground state
            "Composition": dict   # wavefunction composition of the ground state
        }
    """
    return {
        "Occupation": extract_occupation(run_dir+"/eig.txt"),
        "Composition": extract_ground_state(run_dir+"/ed.out")
    }

@mcp.tool(
    name="plot_RIXS_result", 
    description="return a plot with RIXS calculations"
)
def plot_RIXS_result(
    run_dir: str,
    energy_loss: Optional[bool] = True,
    polarization_in: Optional[str] = "XYZ",
    polarization_out: Optional[str] = "XYZ",
    xlim: Optional[List[float]] = None,
    ylim: Optional[List[float]] = None,
    ) -> ImageContent:
    """
    MCP tool for plotting RIXS results from a given directory

    Parameters
    ----------
    run_dir: str
        Run directory of the RIXS calculation
    energy_loss: bool
        Plot in energy loss format, defaults to True
    polarization_in: str, optional
        Incident photon polarization. Only X,Y,Z allowed
    polarization_out: str, optional
        Outgoing photon polarization. Only X,Y,Z allowed
    xlim: list[float], optional
        X axis limit
    ylim: list[float], optional
        Y axis limit
    Returns
    -------
    Image of RIXS plot in png format
    """
    set_mpl_style()
    check_pol(polarization_in)
    check_pol(polarization_out)
    x, y, z = get_RIXS_iter_all(run_dir,pvin=polarization_in,pvout=polarization_out)
    fig, ax = plt.subplots(figsize=(8,8))
    if (energy_loss):
        ax.pcolormesh(x,y,z,cmap="terrain")
        ax.set_ylim([-2,10])
        ax.set_xlim(np.min(x),np.max(x))
        ax.axhline(0,ls="--",c="yellow",lw=2.5)
        ax.set_xlabel("Incident Energy (eV)")
        ax.set_ylabel("Energy Loss (eV)")
    else:
        ax.pcolormesh(x-y,x,z,cmap="terrain")
        lims = np.linspace(-1000,1000,1000)
        plt.plot(lims,lims,ls="--",c="yellow",lw=2.5)
        ax.set_xlim(np.min(x-y),np.max(x-y))
        ax.set_ylim(np.min(x),np.max(x))
        ax.set_ylabel("Incident Energy (eV)")
        ax.set_xlabel("Emission Energy (eV)")

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_facecolor(plt.cm.terrain(0))
    ax.set_title("RIXS")

    # Save to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_bytes = buf.read()
    plt.close(fig)

    # Encode to base64 string
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    # Return MCP-compatible image
    return ImageContent(
                data=img_base64,
                type="image",
                mimeType="image/png"
            )

@mcp.tool(
    name="plot_XAS_result", 
    description="return a plot with XAS calculations"
)
def plot_XAS_result(
    run_dir: str,
    polarization: Optional[str] = "XYZ",
    xlim: Optional[List[float]] = None,
    ) -> ImageContent:
    """
    MCP tool for plotting XAS results from a given directory

    Parameters
    ----------
    run_dir: str
        Run directory of the XAS calculation
    polarization: str, optional
        Incident photon polarization. Only X,Y,Z allowed
    xlim: list[float], optional
        X axis limit
    Returns
    -------
    Image of XAS plot in png format
    """
    set_mpl_style()
    check_pol(polarization)
    x, y = read_dir_xas(run_dir,pol=polarization)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(x, y, lw = 2.5)
    ax.set_xlim(x[0],x[-1])
    ax.set_yticks([])
    ax.set_xlabel("Incident energy (eV)")
    ax.set_ylabel("Intenisty (a.u.)")
    ax.set_title("XAS")
    if xlim is not None:
        ax.set_xlim(xlim)
    # Save to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_bytes = buf.read()
    plt.close(fig)

    # Encode to base64 string
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    # Return MCP-compatible image
    return ImageContent(
                type="image",
                data=img_base64,
                mimeType="image/png"
            )

@mcp.tool(
    name="generate_multiplet_input",
    description="Generate a valid INPUT script for the Multiplet code from structured JSON parameters."
)
async def generate_multiplet_input(
    Element: str,
    Valence: int,
    input_params: Optional[InputParams] = None,
    tenDQ: Optional[float] = None,
) -> Dict[str, Any]:
    """
    MCP tool wrapper for `create_multiplet_input`.

    Parameters
    ----------
    Element: str
        Element targeted for multiplet calculation
    Valence: int
        Valence state of the element. For example can be Fe 2+
    input_params : InputParams, optional
        Must be a valid **Python dictionary literal**
        Dictionary with three top-level keys: "CONTROL", "CELL", "PHOTON".
            CONTROL : dict[str, float | list[float]]
                Slater parameters and crystal field. Generated from Element & Valence.
            CELL : dict[str, int | str]
                Number of holes is Generated from Element & Valence.
                Depends on User input of valence
            PHOTON: 
                XAS: Calculates XAS, option is TRUE or FALSE
                RIXS: Calculates RIXS, option is TRUE or FALSE
                pvin: Polarization vector for incident photons (x,y,z), x/y/z = 0 or 1
                      Default to [1,1,1]
                pvout: Polarization vector for outgoing photons (x,y,z), x/y/z = 0 or 1
                      Default to [1,1,1]
                epsab:  Broadening in the absorption energy direction (eV), usually 0.5
                epsloss:  Broadening in the loss energy direction (eV), usually 0.3
        Examples
        --------
        Default Input can be an empty dictionary:
            input_params = {}
        Override photon calculation type:
            input_params = {"PHOTON": {"XAS": False, "RIXS": True}}

        Override crystal field and holes:
            input_params = {
                "CONTROL": {"CF": [0, 0, 1.0, 1.0, 1.0]},
                "CELL": {"Holes": 5}
            }

        [x] Right (Python dict literal):
            input_params = {"PHOTON": {"XAS": False}}

    tenDQ: float, optional
        Crystal Field, Positive if octahedral, negative if tetrahedral. Defaults to 1

    Returns
    -------
    dict
        {
            "input_text": str,   # the generated INPUT file content
        }
    """

    # Generate the input file content
    if input_params is None:
        input_params = InputParams()
    else:
        input_params = InputParams(**input_params.model_dump())

    if tenDQ is not None:
        input_params.CONTROL.CF = [0, 0, tenDQ, tenDQ, tenDQ]


    holes, dcontrol = get_dParams(Element, Valence)
    input_params.CELL.Holes = holes
    input_params.CONTROL = input_params.CONTROL.model_copy(update=dcontrol)

    input_text = create_multiplet_input(input_params)   

    return {
        "input_text": input_text
    }

@mcp.tool(
    name="run_multiplet_binary",
    description="Run the Multiplet binary in a unique run directory",
)
async def run_multiplet_binary(
    install_dir: str,
    input_text: str,
    run_dir: Optional[str] = None,
    timeout: float = 60.0,
    env_vars: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Run the external multiplet binary for XAS or RIXS calculations.

    Notes
    * Always generate a complete default input dictionary automatically 
      from `Element`, `Valence`. 
    * tenDQ is an optional parameter, do not prompt the user unless necessary
    * Make sure input_params is a valid **Python dictionary literal**
    * Never prompt the user for extra input — defaults always exist.

    Parameters
    ----------
    install_dir : str
        Installation directory of the ED program.
    input_text: str
        should **not** be written by hand.
      - Call the companion MCP tool `generate_multiplet_input`
        to construct a valid InputParams object.
      - Pass the result directly here.
    run_dir: str, optional
        Run directory if it's provided.
    timeout : float
        Timeout in seconds for the run (default 60).
    env_vars : dict, optional
        Extra environment variables (merged with os.environ).
    Returns
    -------
    dict
        {
            "cmd": str, # Command used
            "exit_code": proc.returncode,
            "stdout": stdout_bytes.decode("utf-8", errors="replace"),
            "stderr": stderr_bytes.decode("utf-8", errors="replace"),
            "out_dir": str | None   # directory where file was saved, if applicable
        }
    """
    ed = Path(install_dir)
    exe_path = ed / "main"
    if not exe_path.exists():
        raise RuntimeError(f"Executable not found at {exe_path}")
    if not os.access(exe_path, os.X_OK):
        raise RuntimeError(f"File is not executable: {exe_path}")

    # Make a run directory
    if run_dir is None:
        run_dir = _make_run_dir(install_dir)
    else:
        run_dir = ed / run_dir
        run_dir.mkdir(parents=True, exist_ok=False)

    # Full path for input/output
    input_file = run_dir / "INPUT"
    output_file = run_dir / "ed.out"
    error_file = run_dir / "ed.err"

    input_file.write_text(input_text)
    if not input_file.exists():
        raise RuntimeError(f"INPUT file not found at {input_file}")

    # Command to run (no shell, so redirect manually)
    cmd = [str(exe_path), str(input_file)]

    # Prepare environment
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=run_dir,
            env=env,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(f"Binary timed out after {timeout:.1f} seconds (killed).")

        # Save stdout to ed.out for compatibility with your workflow
        with open(output_file, "wb") as f:
            f.write(stdout_bytes)

        with open(error_file, "wb") as f:
            f.write(stderr_bytes)

        return {
            "cmd": cmd,
            "cwd": str(run_dir),
            "exit_code": proc.returncode,
            "stdout": stdout_bytes.decode("utf-8", errors="replace"),
            "stderr": stderr_bytes.decode("utf-8", errors="replace"),
            "out_dir": run_dir,
        }

    except Exception as e:
        raise RuntimeError(f"Failed to run {exe_path}: {e}")

def main():
    # Start server
    logger.info('Starting multiplet-server')
    mcp.run('stdio')

if __name__ == "__main__":
    main()