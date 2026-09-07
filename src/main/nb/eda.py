import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors

class SensorEDA:

    def __init__(self,ds:xr.Dataset,ref_var=None,entity_dim="sensor"):

        self.ds = ds
        self.ref_var = ref_var
        self.entity_dim = entity_dim

        self.C_LIGHT = "#7979FF"
        self.C_BOLD = "#0000FF"
        self.C_DEEP_BLUE = "#000082"
        self.cmap_custom = mcolors.LinearSegmentedColormap.from_list("custom_blue", ["#FFFFFF", "#0000FF", "#000082"]
)

    # =========================
    # INTERNAL HELPERS
    # =========================
    def _get_ds(self):
        return getattr(self, "ds_clean", self.ds)
    
    def _is_single_entity(self, ds=None):
        ds = ds or self._get_ds()
        return self.entity_dim not in ds.dims

    def _has_entity_dim(self, ds=None):
        ds = ds or self._get_ds()
        return self.entity_dim in ds.dims

    def _get_entities(self, ds=None):
        ds = ds or self._get_ds()

        if self._has_entity_dim(ds):
            vals = list(ds[self.entity_dim].values)
            return [str(v) for v in vals]
        return [None]

    def _select_entity(self, ds, entity):
        if self._has_entity_dim(ds):
            if entity is None:
                return ds
            return ds.sel({self.entity_dim: entity})
        return ds

    def _is_numeric(self, da):
        return np.issubdtype(da.dtype, np.number)

    def _is_time_series_var(self, da):
        return "time" in da.dims and self._is_numeric(da)

    def _flatten_values(self, da):
        arr = da.values.astype("float64")
        return arr.ravel()

    def _get_ref_var(self, ds=None):
        ds = ds or self._get_ds()

        if self.ref_var is not None:
            if self.ref_var not in ds.data_vars:
                raise ValueError(f"ref_var='{self.ref_var}' not found in dataset.")
            return self.ref_var

        # fallback automático: primera variable numérica con dimensión time
        for v in ds.data_vars:
            if self._is_time_series_var(ds[v]):
                return v

        raise ValueError("No numeric time-series variable found to use as reference.")
    

    # =========================
    # PLOT HELPERS
    # =========================
    def _pretty_name(self, var: str, mapping=None) -> str:
        if mapping is None:
            return var
        return mapping.get(var, var)

    def _format_time_axis(self, ax):
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.grid(True, which="major", alpha=0.25)

    def _get_valid_plot_vars(self, ds, only_1d=False):
        valid = []

        for v in ds.data_vars:
            da = ds[v]

            if not self._is_time_series_var(da):
                continue

            if only_1d and da.ndim != 1:
                continue

            if np.isnan(da.values.astype("float64")).all():
                continue

            valid.append(v)

        return valid

    def _extract_series(self, ds_e, var):
        da = ds_e[var]

        if da.ndim == 1:
            s = pd.Series(da.values.astype("float64"), index=pd.to_datetime(ds_e.time.values))
            return s.dropna()

        # si por algún motivo sigue siendo 2D, intentamos colapsar (fallback)
        arr = da.values.astype("float64")
        if arr.ndim == 2:
            arr = arr[:, 0]
            s = pd.Series(arr, index=pd.to_datetime(ds_e.time.values))
            return s.dropna()

        return pd.Series(dtype="float64")
    
    def _aggregate_series(self, series, freq="1D", agg="mean"):
        if series.empty:
            return series

        if agg == "sum":
            return series.resample(freq).sum(min_count=1)

        return series.resample(freq).mean()

    # =========================
    # 1. CLEANING
    # =========================
    def clean_dataset(self):
        ds = self.ds

        # Parse time
        time = pd.to_datetime(ds["time"].values, errors="coerce")
        valid = ~pd.isna(time)

        ds = ds.isel(time=valid)

        # Sort
        ds = ds.sortby("time")

        # Remove duplicates
        t = pd.to_datetime(ds["time"].values)
        dup_mask = pd.Series(t).duplicated(keep="first").values

        if dup_mask.any():
            ds = ds.isel(time=~xr.DataArray(dup_mask, dims="time"))

        self.ds_clean = ds

        return ds

    # =========================
    # 2. SUMMARY
    # =========================
    def dataset_summary(self):
        ds = self._get_ds()

        summary = {
            "n_vars": len(ds.data_vars),
            "n_time": len(ds["time"]),
            "entity_dim": self.entity_dim if self._has_entity_dim(ds) else None,
            "n_entities": ds.sizes.get(self.entity_dim, 1),
            "time_start": str(pd.to_datetime(ds.time.values[0])),
            "time_end": str(pd.to_datetime(ds.time.values[-1])),
        }

        return summary

    def dataset_structure(self):
        ds = self._get_ds()

        return {
            "dims": dict(ds.sizes),
            "coords": list(ds.coords),
            "data_vars": list(ds.data_vars),
        }

    # =========================
    # 3. SENSOR INVENTORY
    # =========================
    def entity_inventory(self):
        ds = self._get_ds()
        ref_var = self._get_ref_var(ds)

        entities = self._get_entities(ds)
        rows = []

        for i, ent in enumerate(entities):
            ds_e = self._select_entity(ds, ent)

            da = ds_e[ref_var]
            values = da.values.astype("float64")

            valid_mask = ~np.isnan(values)
            valid_idx = np.where(valid_mask)[0]

            if len(valid_idx) == 0:
                rows.append({
                    "entity": str(ent),
                    "entity_idx": i,
                    "data_start": pd.NaT,
                    "data_end": pd.NaT,
                    "count": 0,
                    "status": "all_missing"
                })
                continue

            start = pd.to_datetime(ds_e.time.values[valid_idx[0]])
            end = pd.to_datetime(ds_e.time.values[valid_idx[-1]])

            rows.append({
                "entity": str(ent),
                "entity_idx": i,
                "data_start": start,
                "data_end": end,
                "count": int(valid_mask.sum()),
                "status": "ok"
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("data_start").reset_index(drop=True)

        return df
    

    # =========================
    # 4. GLOBAL CADENCE + GAPS
    # =========================
    def global_cadence(self, gap_threshold_hours=3):

        ds = self._get_ds()

        t = pd.to_datetime(ds.time.values)
        dt = pd.Series(t).diff().dropna()

        cadence_seconds = dt.dt.total_seconds()

        cadence_desc = cadence_seconds.describe(
            percentiles=[0.5, 0.9, 0.99]
        ).to_frame("seconds")

        cadence_desc["timedelta"] = pd.to_timedelta(
            cadence_desc["seconds"], unit="s"
        ).astype(str)

        return cadence_desc

    def global_gaps(self, gap_threshold_hours=3, top_n=20):

        ds = self._get_ds()

        gap_threshold = pd.Timedelta(hours=gap_threshold_hours)

        t = pd.to_datetime(ds.time.values)

        # index datetime (CLAVE)
        dt = pd.Series(t, index=t).diff().dropna()

        gaps_global = dt[dt > gap_threshold].sort_values(ascending=False)

        gaps_global_df = pd.DataFrame({
            "gap": gaps_global.values,
            "gap_str": gaps_global.astype(str).values,
            "time_after_gap": gaps_global.index,
            "time_before_gap": gaps_global.index - gaps_global.values,
        }).head(top_n)

        return gaps_global_df

    # =========================
    # 5. GAPS POR SENSOR
    # =========================
    def entity_gaps(self, gap_threshold_hours=2):
        ds = self._get_ds()
        ref_var = self._get_ref_var(ds)

        gap_threshold = pd.Timedelta(hours=gap_threshold_hours)
        rows = []

        for ent in self._get_entities(ds):
            ds_e = self._select_entity(ds, ent)

            series = pd.Series(
                ds_e[ref_var].values.astype("float64"),
                index=pd.to_datetime(ds_e.time.values)
            )

            valid = series.dropna()

            if valid.empty:
                rows.append({
                    "entity": str(ent),
                    "largest_gap": pd.NaT,
                    "gap_start": pd.NaT,
                    "gap_end": pd.NaT,
                    "mean_gap": pd.NaT,
                    "n_gaps": 0,
                    "status": "all_missing"
                })
                continue

            dt = valid.index.to_series().diff().dropna()
            gaps = dt[dt > gap_threshold]

            if gaps.empty:
                rows.append({
                    "entity": str(ent),
                    "largest_gap": pd.Timedelta(0),
                    "gap_start": pd.NaT,
                    "gap_end": pd.NaT,
                    "mean_gap": pd.Timedelta(0),
                    "n_gaps": 0,
                    "status": "ok"
                })
            else:
                largest_gap_idx = gaps.idxmax()
                largest_gap = gaps.max()

                rows.append({
                    "entity": str(ent),
                    "largest_gap": largest_gap,
                    "gap_start": largest_gap_idx - largest_gap,
                    "gap_end": largest_gap_idx,
                    "mean_gap": gaps.mean(),
                    "n_gaps": len(gaps),
                    "status": "ok"
                })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("largest_gap", ascending=False)

        return df

    # =========================
    # 6. MISSINGNESS GLOBAL
    # =========================
    def missingness_global(self):

        ds = self._get_ds()

        missing_rows = []

        for v in ds.data_vars:

            da = ds[v]

            if not self._is_time_series_var(da):
                continue

            arr = self._flatten_values(da)

            n = arr.size
            n_missing = np.isnan(arr).sum()

            missing_rows.append({
                "variable": v,
                "missing_count": int(n_missing),
                "missing_pct": float(n_missing / n * 100),
            })

        missing_df = pd.DataFrame(missing_rows).sort_values(
            ["missing_pct", "variable"],
            ascending=[False, True]
        )

        return missing_df

    # =========================
    # 7. STATS GLOBAL
    # =========================
    def stats_global(self):

        ds = self._get_ds()

        var_rows = []

        for v in ds.data_vars:

            da = ds[v]

            if not self._is_time_series_var(da):
                continue

            arr = self._flatten_values(da)
            n_missing = np.isnan(arr).sum()
            n = arr.size

            s = pd.Series(arr.flatten())

            var_rows.append({
                "variable": v,
                "dtype": str(da.dtype),
                "min": float(np.nanmin(arr)) if n_missing < n else np.nan,
                "p01": float(s.quantile(0.01)) if n_missing < n else np.nan,
                "p05": float(s.quantile(0.05)) if n_missing < n else np.nan,
                "median": float(np.nanmedian(arr)) if n_missing < n else np.nan,
                "mean": float(np.nanmean(arr)) if n_missing < n else np.nan,
                "std": float(np.nanstd(arr)) if n_missing < n else np.nan,
                "p95": float(s.quantile(0.95)) if n_missing < n else np.nan,
                "p99": float(s.quantile(0.99)) if n_missing < n else np.nan,
                "max": float(np.nanmax(arr)) if n_missing < n else np.nan,
            })

        stats_df = pd.DataFrame(var_rows).sort_values("variable")

        return stats_df
    
    # =========================
    # 8. MISSINGNESS + STATS POR SENSOR
    # =========================
    def entity_variable_stats(self):
        ds = self._get_ds()
        entities = self._get_entities(ds)

        rows = []

        for v in ds.data_vars:
            da = ds[v]

            if not self._is_time_series_var(da):
                continue

            # Caso multi-entity: la variable tiene entity_dim
            if self._has_entity_dim(ds) and self.entity_dim in da.dims:
                for ent in entities:
                    arr = da.sel({self.entity_dim: ent}).values.astype("float64")

                    n = arr.size
                    n_missing = np.isnan(arr).sum()

                    if n_missing < n:
                        s = pd.Series(arr)
                        row = {
                            "entity": str(ent),
                            "variable": v,
                            "missing_count": int(n_missing),
                            "missing_pct": float(n_missing / n * 100),
                            "min": float(np.nanmin(arr)),
                            "p01": float(s.quantile(0.01)),
                            "p05": float(s.quantile(0.05)),
                            "median": float(np.nanmedian(arr)),
                            "mean": float(np.nanmean(arr)),
                            "std": float(np.nanstd(arr)),
                            "p95": float(s.quantile(0.95)),
                            "p99": float(s.quantile(0.99)),
                            "max": float(np.nanmax(arr)),
                        }
                    else:
                        row = {
                            "entity": str(ent),
                            "variable": v,
                            "missing_count": int(n_missing),
                            "missing_pct": 100.0,
                            "min": np.nan,
                            "p01": np.nan,
                            "p05": np.nan,
                            "median": np.nan,
                            "mean": np.nan,
                            "std": np.nan,
                            "p95": np.nan,
                            "p99": np.nan,
                            "max": np.nan,
                        }

                    rows.append(row)

            # Caso single-entity: variable solo depende de time
            else:
                arr = da.values.astype("float64")
                n = arr.size
                n_missing = np.isnan(arr).sum()

                if n_missing < n:
                    s = pd.Series(arr)
                    row = {
                        "entity": "__single_entity__",
                        "variable": v,
                        "missing_count": int(n_missing),
                        "missing_pct": float(n_missing / n * 100),
                        "min": float(np.nanmin(arr)),
                        "p01": float(s.quantile(0.01)),
                        "p05": float(s.quantile(0.05)),
                        "median": float(np.nanmedian(arr)),
                        "mean": float(np.nanmean(arr)),
                        "std": float(np.nanstd(arr)),
                        "p95": float(s.quantile(0.95)),
                        "p99": float(s.quantile(0.99)),
                        "max": float(np.nanmax(arr)),
                    }
                else:
                    row = {
                        "entity": "__single_entity__",
                        "variable": v,
                        "missing_count": int(n_missing),
                        "missing_pct": 100.0,
                        "min": np.nan,
                        "p01": np.nan,
                        "p05": np.nan,
                        "median": np.nan,
                        "mean": np.nan,
                        "std": np.nan,
                        "p95": np.nan,
                        "p99": np.nan,
                        "max": np.nan,
                    }

                rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(
                ["missing_pct", "variable", "entity"],
                ascending=[False, True, True]
            )

        return df

    # =========================
    # 9. MISSING DURANTE VIDA ÚTIL
    # =========================
    def entity_lifecycle_missing(self):
        ds = self._get_ds()

        rows = []

        for ent in self._get_entities(ds):
            ds_e = self._select_entity(ds, ent)

            for v in ds.data_vars:
                da = ds_e[v]

                if not self._is_time_series_var(da):
                    continue

                values = da.values.astype("float64")
                valid_idx = np.where(~np.isnan(values))[0]

                if len(valid_idx) == 0:
                    continue

                first_idx = valid_idx[0]
                last_idx = valid_idx[-1]

                da_active = da.isel(time=slice(first_idx, last_idx + 1))

                n_total = da_active.size
                n_missing = np.isnan(da_active.values).sum()

                rows.append({
                    "entity": str(ent),
                    "variable": v,
                    "missing_during_lifecycle": int(n_missing),
                    "pct_missing_lifecycle": float(n_missing / n_total * 100),
                    "total_points": int(n_total),
                    "from": pd.to_datetime(da_active.time.values[0]),
                    "to": pd.to_datetime(da_active.time.values[-1]),
                })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("pct_missing_lifecycle", ascending=False)

        return df
    
    # =========================
    # 10. MISSING CONDICIONAL
    # =========================
    def conditional_missing(self):

        ds = self._get_ds()

        # seleccionar variables válidas
        optional_vars = [
            v for v in ds.data_vars
            if self._is_time_series_var(ds[v])
        ]

        rows = []

        for v in optional_vars:

            da = ds[v]

            # resto de variables
            other_vars = [vv for vv in optional_vars if vv != v]

            others_have_data = np.zeros_like(da.values, dtype=bool)

            for vv in other_vars:
                other_arr = ds[vv].values.astype("float64")
                others_have_data |= ~np.isnan(other_arr)

            # esta variable missing
            this_missing = np.isnan(da.values)

            # casos problemáticos
            bad = others_have_data & this_missing

            total = others_have_data.sum()

            rows.append({
                "variable": v,
                "missing_when_others_present": int(bad.sum()),
                "total_when_others_present": int(total),
                "missing_pct_conditional": (
                    float(bad.sum() / total * 100)
                    if total > 0 else np.nan
                )
            })

        missing_cond_df = pd.DataFrame(rows).sort_values(
            "missing_pct_conditional",
            ascending=False
        )

        return missing_cond_df
    
    def plot_timeseries(
        self,
        vars_to_plot=None,
        freq="1D",
        rolling_window=7,
        mapping_names=None,
        agg_map=None,
        per_entity=True,
        max_entities=None,
        figsize_base=(12, 2.8),
        dpi=130
    ):

        plt.style.use("seaborn-v0_8-whitegrid")

        ds = self._get_ds()

        if vars_to_plot is None:
            # default: todas las variables temporales numéricas válidas
            vars_to_plot = self._get_valid_plot_vars(ds)

        if not vars_to_plot:
            print("[plot_timeseries] No valid variables found to plot.")
            return

        entities = self._get_entities(ds)

        if max_entities is not None:
            entities = entities[:max_entities]

        # si no hay entity_dim, solo una figura total
        if not self._has_entity_dim(ds):
            per_entity = False

        if agg_map is None:
            agg_map = {}

        for ent in entities if per_entity else [None]:

            ds_e = self._select_entity(ds, ent)

            nvars = len(vars_to_plot)
            fig, axes = plt.subplots(
                nvars, 1,
                figsize=(figsize_base[0], figsize_base[1] * nvars),
                dpi=dpi,
                sharex=True
            )

            if nvars == 1:
                axes = [axes]

            for i, v in enumerate(vars_to_plot):
                if v not in ds_e:
                    continue

                ax = axes[i]

                s = self._extract_series(ds_e, v)
                if s.empty:
                    ax.set_ylabel(v)
                    ax.set_title(f"{v} (no data)")
                    continue

                agg = agg_map.get(v, "mean")

                daily = self._aggregate_series(s, freq=freq, agg=agg)
                smooth = daily.rolling(window=rolling_window, min_periods=3).mean()

                ax.plot(daily.index, daily.values, color=self.C_LIGHT, linewidth=1.2, alpha=0.7, label=f"{freq} {agg}")
                ax.plot(smooth.index, smooth.values, color=self.C_BOLD, linewidth=2.3, label=f"{rolling_window}-window mean")

                ymin = np.nanmin(s.values)
                ymax = np.nanmax(s.values)

                if np.isfinite(ymin) and np.isfinite(ymax):
                    if ymin == ymax:
                        ax.set_ylim(ymin - 1, ymax + 1)
                    else:
                        pad = (ymax - ymin) * 0.05
                        ax.set_ylim(ymin - pad, ymax + pad)

                ax.set_ylabel(self._pretty_name(v, mapping_names))
                self._format_time_axis(ax)

                if i == 0:
                    ax.legend(loc="best", frameon=True)

            title = f"{self.entity_dim} {ent} — Timeseries" if per_entity else "Timeseries"
            axes[0].set_title(title, fontweight="bold", pad=10)

            plt.tight_layout()
            plt.show()

    def plot_distributions(
        self,
        vars_to_plot=None,
        mapping_names=None,
        clip_quantiles=(0.01, 0.99),
        bins=50,
        per_entity=False,
        max_entities=None,
        dpi=130
    ):

        plt.style.use("seaborn-v0_8-whitegrid")

        ds = self._get_ds()

        if vars_to_plot is None:
            vars_to_plot = self._get_valid_plot_vars(ds)

        if not vars_to_plot:
            print("[plot_distributions] No valid variables found to plot.")
            return

        entities = self._get_entities(ds)

        if max_entities is not None:
            entities = entities[:max_entities]

        if not self._has_entity_dim(ds):
            per_entity = False

        ent_loop = entities if per_entity else [None]

        for ent in ent_loop:

            ds_e = self._select_entity(ds, ent)

            for v in vars_to_plot:
                if v not in ds_e:
                    continue

                s = self._extract_series(ds_e, v)
                if s.empty:
                    continue

                lo, hi = s.quantile(list(clip_quantiles)).values
                s_clip = s[(s >= lo) & (s <= hi)]

                if s_clip.empty:
                    continue

                fig, ax = plt.subplots(figsize=(8.5, 3.6), dpi=dpi)
                ax.set_facecolor("#F9F9F9")

                n, bins_arr, patches = ax.hist(
                    s_clip.values,
                    bins=bins,
                    edgecolor="black",
                    linewidth=0.6
                )

                max_h = n.max() if len(n) > 0 else 1
                for val, patch in zip(n, patches):
                    ratio = val / max_h if max_h > 0 else 0
                    patch.set_facecolor(self.cmap_custom(ratio))

                title_var = self._pretty_name(v, mapping_names)
                title = f"{title_var} — Distribution"
                if per_entity:
                    title = f"{self.entity_dim} {ent} — {title_var} — Distribution"

                ax.set_title(title, fontweight="bold")
                ax.grid(True, alpha=0.1)

                plt.tight_layout()
                plt.show()

    def plot_correlation_matrix(
        self,
        vars_to_plot=None,
        mapping_names=None,
        freq="1D",
        agg="median",
        min_non_nan_frac=0.5,
        entity=None,
        dpi=130,
        figsize=(9, 8)
    ):

        plt.style.use("seaborn-v0_8-whitegrid")

        ds = self._get_ds()

        # Selección de variables válidas
        if vars_to_plot is None:
            vars_to_plot = self._get_valid_plot_vars(ds)

        if not vars_to_plot:
            print("[plot_correlation_matrix] No valid variables to plot.")
            return

        # Si es multisensor, elegir entidad concreta
        if self._has_entity_dim(ds):
            entities = self._get_entities(ds)
            if entity is None:
                entity = entities[0]  # default: primera entidad
            ds_e = self._select_entity(ds, entity)
        else:
            ds_e = ds

        # Construir dataframe temporal
        df = ds_e[vars_to_plot].to_dataframe()
        df = df.select_dtypes(include=[np.number])

        if "time" in df.columns:
            df = df.set_index("time")

        df = df.sort_index()

        # Agregación temporal
        if agg == "mean":
            df_res = df.resample(freq).mean()
        elif agg == "sum":
            df_res = df.resample(freq).sum(min_count=1)
        else:
            df_res = df.resample(freq).median()

        # Filtrar variables con suficiente información
        good_vars = []
        for c in df_res.columns:
            frac_non_nan = df_res[c].notna().mean()
            if frac_non_nan >= min_non_nan_frac:
                good_vars.append(c)

        if len(good_vars) < 2:
            print("[plot_correlation_matrix] Not enough variables with sufficient data.")
            return

        corr = df_res[good_vars].corr()

        # Pretty names
        cols = [self._pretty_name(c, mapping_names) for c in corr.columns]
        corr.columns = cols
        corr.index = cols

        # Plot
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        im = ax.imshow(corr.values, cmap=self.cmap_custom, vmin=-1, vmax=1)

        ax.set_xticks(np.arange(len(cols)))
        ax.set_yticks(np.arange(len(cols)))

        ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(cols, fontsize=10)

        ax.grid(False)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.outline.set_linewidth(0.5)

        title = "CORRELATION MATRIX"
        if self._has_entity_dim(ds):
            title = f"CORRELATION MATRIX — {self.entity_dim} {entity}"

        ax.set_title(title, pad=20, fontweight="bold")

        plt.tight_layout()
        plt.show()

    def plot_availability(
        self,
        var=None,
        freq="1D",
        per_entity=False,
        max_entities=50,
        dpi=130,
        figsize=(12, 3.2)
    ):

        plt.style.use("seaborn-v0_8-whitegrid")

        ds = self._get_ds()

        if var is None:
            var = self._get_ref_var()

        if var not in ds:
            valid = self._get_valid_plot_vars(ds)
            if not valid:
                print("[plot_availability] No valid vars found.")
                return
            var = valid[0]

        # Caso monosensor
        if not self._has_entity_dim(ds):

            s = self._extract_series(ds, var)
            if s.empty:
                print("[plot_availability] No data for availability.")
                return

            avail = s.notna().resample(freq).mean()

            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            ax.plot(avail.index, avail.values, color=self.C_DEEP_BLUE, linewidth=2)

            ax.set_title("Availability", fontweight="bold")
            ax.set_ylabel("fraction available")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            self._format_time_axis(ax)

            plt.tight_layout()
            plt.show()
            return

        # Caso multisensor
        entities = self._get_entities(ds)

        if max_entities is not None:
            entities = entities[:max_entities]

        # Extraer DataFrame (time x entity)
        da = ds[var]

        # to_pandas para tener tabla
        df = da.to_pandas()

        # Asegurar índice temporal
        df = df.sort_index()

        avail = df.notna().resample(freq).mean()

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        if per_entity:
            for col in avail.columns:
                ax.plot(avail.index, avail[col], color=self.C_LIGHT, linewidth=0.8, alpha=0.35)
            mean_avail = avail.mean(axis=1)
            ax.plot(mean_avail.index, mean_avail.values, color=self.C_DEEP_BLUE, linewidth=2.5, label="mean availability")
            ax.legend()
        else:
            mean_avail = avail.mean(axis=1)
            ax.plot(mean_avail.index, mean_avail.values, color=self.C_DEEP_BLUE, linewidth=2.5)

        ax.set_title("Availability", fontweight="bold")
        ax.set_ylabel("fraction active sensors")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        self._format_time_axis(ax)

        plt.tight_layout()
        plt.show()
    
    def plot_availability_heatmap(
        self,
        var=None,
        freq="1D",
        max_entities=80,
        dpi=130,
        figsize=(14, 8)
    ):
        """
        Heatmap de disponibilidad: filas=sensores, columnas=tiempo (agregado por freq).
        """

        plt.style.use("seaborn-v0_8-whitegrid")

        ds = self._get_ds()

        if not self._has_entity_dim(ds):
            print("[plot_availability_heatmap] Dataset has no entity_dim. Use plot_availability instead.")
            return

        if var is None:
            var = self._get_ref_var(ds)

        if var not in ds:
            print(f"[plot_availability_heatmap] var='{var}' not found in dataset.")
            return

        entities = self._get_entities(ds)[:max_entities]

        da = ds[var].sel({self.entity_dim: entities})
        df = da.to_pandas()  # index=time, columns=entities
        df = df.sort_index()

        avail = df.notna().resample(freq).mean()  # time x sensor
        mat = avail.T.values  # sensor x time

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        im = ax.imshow(mat, aspect="auto", cmap=self.cmap_custom, vmin=0, vmax=1)

        ax.set_yticks(np.arange(len(entities)))
        ax.set_yticklabels([str(e) for e in entities], fontsize=8)

        # ticks de tiempo reducidos
        ncols = avail.shape[0]
        step = max(1, ncols // 10)
        xticks = np.arange(0, ncols, step)

        ax.set_xticks(xticks)
        ax.set_xticklabels(
            [avail.index[i].strftime("%Y-%m-%d") for i in xticks],
            rotation=45,
            ha="right",
            fontsize=9
        )

        ax.set_title(f"Availability Heatmap — {var} ({freq})", fontweight="bold", pad=15)
        ax.set_xlabel("time")
        ax.set_ylabel(self.entity_dim)

        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("availability fraction")

        plt.tight_layout()
        plt.show()

    def plot_variable_overlay(
        self,
        var,
        freq="1D",
        agg="mean",
        rolling_window=7,
        max_entities=80,
        alpha_sensors=0.15,
        dpi=130,
        figsize=(14, 4)
    ):
        """
        Plot por variable: todas las series de sensores en gris claro + mediana global.
        """

        plt.style.use("seaborn-v0_8-whitegrid")

        ds = self._get_ds()

        if var not in ds:
            print(f"[plot_variable_overlay] var='{var}' not found in dataset.")
            return

        if not self._has_entity_dim(ds):
            print("[plot_variable_overlay] Dataset has no entity_dim. Use plot_timeseries instead.")
            return

        entities = self._get_entities(ds)[:max_entities]

        da = ds[var].sel({self.entity_dim: entities})
        df = da.to_pandas()  # time x sensor
        df = df.sort_index()

        # agregación temporal
        if agg == "sum":
            df_res = df.resample(freq).sum(min_count=1)
        elif agg == "median":
            df_res = df.resample(freq).median()
        else:
            df_res = df.resample(freq).mean()

        global_median = df_res.median(axis=1)
        global_smooth = global_median.rolling(rolling_window, min_periods=3).mean()

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # sensores individuales
        for col in df_res.columns:
            ax.plot(df_res.index, df_res[col], color="gray", linewidth=0.8, alpha=alpha_sensors)

        # median global
        ax.plot(global_median.index, global_median.values,
                color=self.C_LIGHT, linewidth=2.0, label="global median")

        ax.plot(global_smooth.index, global_smooth.values,
                color=self.C_DEEP_BLUE, linewidth=2.7, label=f"rolling median ({rolling_window})")

        ax.set_title(f"{var} — overlay sensors ({freq} {agg})", fontweight="bold")
        ax.set_ylabel(var)
        ax.legend()
        self._format_time_axis(ax)

        plt.tight_layout()
        plt.show()

    def plot_boxplot_by_entity(
            self,
            var,
            clip_quantiles=(0.01, 0.99),
            max_entities=50,
            dpi=130,
            figsize=(14, 5)
        ):
        """
        Boxplot por sensor para una variable.
        """

        plt.style.use("seaborn-v0_8-whitegrid")

        ds = self._get_ds()

        if var not in ds:
            print(f"[plot_boxplot_by_entity] var='{var}' not found in dataset.")
            return

        if not self._has_entity_dim(ds):
            print("[plot_boxplot_by_entity] Dataset has no entity_dim.")
            return

        entities = self._get_entities(ds)[:max_entities]

        da = ds[var].sel({self.entity_dim: entities})
        df = da.to_pandas()

        stacked = df.stack().dropna()
        if stacked.empty:
            print("[plot_boxplot_by_entity] No data.")
            return

        lo, hi = stacked.quantile(list(clip_quantiles)).values
        df_clip = df.clip(lower=lo, upper=hi)

        data = [df_clip[c].dropna().values for c in df_clip.columns]

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        bp = ax.boxplot(
            data,
            patch_artist=True,
            showfliers=False
        )

        box_color = self.C_LIGHT
        edge_color = self.C_BOLD
        median_color = self.C_DEEP_BLUE
        whisker_color = self.C_BOLD

        # cajas
        for box in bp['boxes']:
            box.set(facecolor=box_color, edgecolor=edge_color, alpha=0.6)

        # medianas
        for median in bp['medians']:
            median.set(color=median_color, linewidth=2)

        # whiskers
        for whisker in bp['whiskers']:
            whisker.set(color=whisker_color, linewidth=1.2)

        # caps
        for cap in bp['caps']:
            cap.set(color=whisker_color, linewidth=1.2)

        ax.set_xticks(np.arange(1, len(entities) + 1))
        ax.set_xticklabels([str(e) for e in entities], rotation=90, fontsize=8)

        ax.set_title(
            f"{var} — distribution by {self.entity_dim}",
            fontweight="bold",
            color=self.C_DEEP_BLUE
        )

        ax.set_ylabel(var, color=self.C_BOLD)
        ax.grid(alpha=0.2)

        plt.tight_layout()
        plt.show()

    def plot_entity_correlation_matrix(
        self,
        var,
        freq="1D",
        agg="mean",
        min_non_nan_frac=0.6,
        max_entities=40,
        dpi=130,
        figsize=(10, 9)
    ):
        """
        Matriz de correlación sensor-sensor para una variable.
        """

        plt.style.use("seaborn-v0_8-whitegrid")

        ds = self._get_ds()

        if var not in ds:
            print(f"[plot_entity_correlation_matrix] var='{var}' not found in dataset.")
            return

        if not self._has_entity_dim(ds):
            print("[plot_entity_correlation_matrix] Dataset has no entity_dim.")
            return

        entities = self._get_entities(ds)[:max_entities]

        da = ds[var].sel({self.entity_dim: entities})
        df = da.to_pandas()
        df = df.sort_index()

        # agregación temporal
        if agg == "sum":
            df_res = df.resample(freq).sum(min_count=1)
        elif agg == "median":
            df_res = df.resample(freq).median()
        else:
            df_res = df.resample(freq).mean()

        # filtrar sensores con suficiente info
        good_cols = []
        for c in df_res.columns:
            frac = df_res[c].notna().mean()
            if frac >= min_non_nan_frac:
                good_cols.append(c)

        if len(good_cols) < 2:
            print("[plot_entity_correlation_matrix] Not enough sensors with sufficient data.")
            return

        corr = df_res[good_cols].corr()

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        im = ax.imshow(corr.values, cmap=self.cmap_custom, vmin=-1, vmax=1)

        labels = [str(c) for c in corr.columns]

        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)

        ax.set_title(f"Sensor-Sensor Correlation — {var} ({freq} {agg})", fontweight="bold", pad=15)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("corr")

        plt.tight_layout()
        plt.show()

    def outlier_summary(
        self,
        vars_to_check=None,
        per_entity=True,
        max_entities=None,
        k=1.5  # factor clásico IQR
    ):
        """
        Outliers reales usando método IQR (robusto).

        Outlier = valor < Q1 - k*IQR o valor > Q3 + k*IQR
        """

        ds = self._get_ds()

        if vars_to_check is None:
            vars_to_check = self._get_valid_plot_vars(ds)

        if not vars_to_check:
            return pd.DataFrame()

        has_entities = self._has_entity_dim(ds)

        if max_entities is not None and has_entities:
            entities = self._get_entities(ds)[:max_entities]
        else:
            entities = self._get_entities(ds)

        rows = []

        for v in vars_to_check:
            if v not in ds:
                continue

            da = ds[v]

            if not self._is_time_series_var(da):
                continue

            # ============================
            # MULTISENSOR
            # ============================
            if has_entities and self.entity_dim in da.dims:

                da_sel = da
                if max_entities is not None:
                    da_sel = da.sel({self.entity_dim: entities})

                if not per_entity:
                    arr = da_sel.values.astype("float64").ravel()
                    arr = arr[~np.isnan(arr)]

                    if arr.size == 0:
                        continue

                    q1 = np.quantile(arr, 0.25)
                    q3 = np.quantile(arr, 0.75)
                    iqr = q3 - q1

                    lo = q1 - k * iqr
                    hi = q3 + k * iqr

                    out_mask = (arr < lo) | (arr > hi)
                    out_count = int(out_mask.sum())
                    out_pct = float(out_count / arr.size * 100)

                    rows.append({
                        "entity": "__all_entities__",
                        "variable": v,
                        "n_points": int(arr.size),
                        "q_low": float(lo),
                        "q_high": float(hi),
                        "outlier_count": out_count,
                        "outlier_pct": out_pct
                    })

                else:
                    for ent in entities:
                        arr = da_sel.sel({self.entity_dim: ent}).values.astype("float64").ravel()
                        arr = arr[~np.isnan(arr)]

                        if arr.size == 0:
                            rows.append({
                                "entity": str(ent),
                                "variable": v,
                                "n_points": 0,
                                "q_low": np.nan,
                                "q_high": np.nan,
                                "outlier_count": np.nan,
                                "outlier_pct": np.nan
                            })
                            continue

                        q1 = np.quantile(arr, 0.25)
                        q3 = np.quantile(arr, 0.75)
                        iqr = q3 - q1

                        lo = q1 - k * iqr
                        hi = q3 + k * iqr

                        out_mask = (arr < lo) | (arr > hi)
                        out_count = int(out_mask.sum())
                        out_pct = float(out_count / arr.size * 100)

                        rows.append({
                            "entity": str(ent),
                            "variable": v,
                            "n_points": int(arr.size),
                            "q_low": float(lo),
                            "q_high": float(hi),
                            "outlier_count": out_count,
                            "outlier_pct": out_pct
                        })

            # ============================
            # MONOSENSOR
            # ============================
            else:
                arr = da.values.astype("float64").ravel()
                arr = arr[~np.isnan(arr)]

                if arr.size == 0:
                    continue

                q1 = np.quantile(arr, 0.25)
                q3 = np.quantile(arr, 0.75)
                iqr = q3 - q1

                lo = q1 - k * iqr
                hi = q3 + k * iqr

                out_mask = (arr < lo) | (arr > hi)
                out_count = int(out_mask.sum())
                out_pct = float(out_count / arr.size * 100)

                rows.append({
                    "entity": "__single_entity__",
                    "variable": v,
                    "n_points": int(arr.size),
                    "q_low": float(lo),
                    "q_high": float(hi),
                    "outlier_count": out_count,
                    "outlier_pct": out_pct
                })

        df = pd.DataFrame(rows)

        if not df.empty:
            df = df.sort_values(
                ["outlier_pct", "variable", "entity"],
                ascending=[False, True, True]
            ).reset_index(drop=True)

        return df