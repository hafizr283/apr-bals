# Large artifacts and dataset access

[Repository entry](../README.md) Â· [Reproduction guide](reproduction.md) Â· [Artifact inventory](artifacts_manifest.json)

The repository keeps source code, CSV results, raw text trajectories, command metadata,
and small research figures in Git. Large datasets, factor arrays, trained models,
compiled executables and caches stay outside Git. Small PDF/PNG figures are valid
repository artifacts; the exclusion is about large generated payloads, not every
file with a binary encoding.

## Available archive

The restored October evidence came from the existing
[new_results.zip on Google Drive](https://drive.google.com/file/d/1c9_HLNKctFwAC5Ts9hofxoWuRiIJM9iZ/view)
([containing folder](https://drive.google.com/drive/folders/1mdVw-JonIiq6gsNs9MZoCURK_6-n7XcC)).
Access depends on the owner's current Drive permissions. This link does not imply
that a public institutional archive or a new dataset redistribution agreement exists.

| Property | Recorded value |
|---|---|
| Filename | `new_results.zip` |
| Compressed bytes | 6,722,845,207 |
| Uncompressed bytes, all members | 17,916,059,206 |
| SHA-256 | `0414eb3fc8cd80cf65b0b1cde7ec0cbfa51237d1bce48f212f69deac1bad3538` |
| Generated/binary/cache files excluded from Git | 322 members; 17,903,911,186 uncompressed bytes |

The [machine-readable inventory](artifacts_manifest.json) records exact excluded
paths, sizes, ZIP CRC-32 values, categories and available archived member hashes.
The whole-ZIP SHA-256 was computed during restoration. Historical per-file hashes
are identified as archived records; every binary has not been independently replayed.

Download to a drive with enough free space. Keeping the ZIP and extracting all its
contents needs the sum of the compressed and uncompressed sizes plus working space.
Use a filesystem supporting files larger than 4 GiB, such as exFAT or NTFS.

Verify the download before using it:

```bash
sha256sum /path/to/new_results.zip
```

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath 'G:\new_results.zip'
```

Compare the result with the exact hash above. Extract into a separate experiment
directory; the full ZIP also contains earlier narrative summaries, so do not use
it to overwrite the repository's curated README or evidence guide.

## What to retain externally

| Artifact | Storage and reproduction treatment |
|---|---|
| Original ratings and prepared splits | Obtain original datasets under their terms. Preserve split manifests and preparation scripts in Git; store permitted binary copies in the external archive. |
| Factors, models and diagnostic arrays | Retain required payloads with exact path, size, checksum and generating command. Their absence from Git does not mean the archived run failed. |
| Compiled CUDA executables | Prefer rebuilding from the restored campaign source and recorded flags. Retain original binaries externally when needed to investigate compiler/runtime provenance. |
| Python caches | Regenerate locally. They are not needed as research evidence. |

The original parent rating binaries and the historical Linux toolchain may be needed
beyond this ZIP. Availability of the bundle alone does not establish clean-machine
reproduction; see the [remaining execution gaps](reproduction.md#what-remains-for-full-reproduction).

## Journal artifact arrangement

For long-term review access, use a versioned institutional, Zenodo or OSF deposit for
redistributable research artifacts and link it to a fixed Git commit. Record the
archive checksum, access conditions, dataset provenance, source/toolchain versions
and preparation commands. Keep restricted datasets behind their original access
process and publish instructions and split metadata instead of unauthorized copies.

The current Drive link is a working source, not a claim of permanent preservation.
A future deposit should retain a stable version and access URL; add a DOI only after
one has actually been issued. No new external upload or release was created in this update.

Git LFS is an option for selected large files when storage and download quotas are
budgeted. It is not required for the CPU evidence check and is not configured here.
For this multi-gigabyte bundle, an external archive plus the small committed manifest
keeps ordinary clones and CI lightweight. Do not commit the ZIP, split binaries,
factor arrays or executables into normal Git history.
