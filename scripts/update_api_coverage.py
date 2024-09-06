import inspect

import galsim

import jax_galsim


def _list_all_apis(module, apis=None):
    apis = apis or set()
    mname = module.__name__
    top_level = mname.split(".")[0]

    for name in dir(module):
        full_name = f"{mname}.{name}"

        if name.startswith("_") or full_name == "jax_galsim.core":
            continue

        obj = getattr(module, name)

        if inspect.ismodule(obj):
            if (
                full_name not in apis
                and (not inspect.isbuiltin(obj))
                and hasattr(obj, "__file__")
                and top_level in obj.__file__.split("/")
            ):
                if not any(api.startswith(f"{full_name}.") for api in apis):
                    apis.add(full_name)
                _list_all_apis(obj, apis=apis)
        elif inspect.isclass(obj) or inspect.isfunction(obj):
            # print(full_name)
            if not any(api.endswith(f".{name}") for api in apis):
                apis.add(full_name)

    return apis


def _write_to_readme(missing_apis, jax_galsim_apis, cov_frac):
    with open("README.md", "r") as f:
        lines = f.readlines()

    start = lines.index("<!-- start-api-coverage -->\n")
    end = lines.index("<!-- end-api-coverage -->\n")

    middle_lines = []
    middle_lines.append(
        f"JAX-GalSim has implemented {100 * cov_frac:.1f}% of "
        "the GalSim API. See the list below for the supported APIs.\n"
    )
    middle_lines.append("\n")

    middle_lines.append("<details>\n")
    middle_lines.append("\n")

    for api in sorted(jax_galsim_apis):
        middle_lines.append(f"- {api}\n")
    middle_lines.append("\n")

    middle_lines.append("</details>\n")

    with open("README.md", "w") as f:
        f.writelines(lines[: start + 1])
        f.writelines(middle_lines)
        f.writelines(lines[end:])


if __name__ == "__main__":
    galsim_apis = _list_all_apis(galsim)
    assert all(api.startswith("galsim.") for api in galsim_apis)

    jax_galsim_apis = _list_all_apis(jax_galsim)
    assert all(api.startswith("jax_galsim.") for api in jax_galsim_apis)

    # make the import prefix match and subset to what is in both
    jax_galsim_apis = {"galsim" + api[len("jax_galsim") :] for api in jax_galsim_apis}
    jax_galsim_apis = galsim_apis & jax_galsim_apis

    missing_apis = galsim_apis - jax_galsim_apis
    assert len(missing_apis) + len(jax_galsim_apis) == len(galsim_apis)

    cov_frac = 1.0 - len(missing_apis) / len(galsim_apis)

    _write_to_readme(missing_apis, jax_galsim_apis, cov_frac)
