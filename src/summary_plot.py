from __future__ import division

import warnings
import numpy as np
import scipy as sp
from scipy.stats import gaussian_kde
import matplotlib.pyplot as pl
from shap.plots._labels import labels
from shap.plots import colors
from shap.utils import safe_isinstance, OpChain, format_value
from shap.plots._utils import convert_ordering, convert_color, merge_nodes, get_sort_order, sort_inds
from shap import Explanation

def summary_legacy(shap_values, features=None, feature_names=None, features_imgPath=None,
                    max_display=None, plot_type=None,
                    color=None, axis_color="#333333", title=None, alpha=1, show=True, sort=True,
                    color_bar=True, plot_size="auto", class_names=None,
                    class_inds=None,
                    custom_order=None,
                    color_bar_label=labels["FEATURE_VALUE"],
                    cmap=colors.red_blue,
                    # depreciated
                    auto_size_plot=None,
                    use_log_scale=False):
    """Create a SHAP beeswarm plot, colored by feature values when they are provided.

    Parameters
    ----------
    shap_values : numpy.array
        For single output explanations this is a matrix of SHAP values (# samples x # features).
        For multi-output explanations this is a list of such matrices of SHAP values.

    features : numpy.array or pandas.DataFrame or list
        Matrix of feature values (# samples x # features) or a feature_names list as shorthand

    feature_names : list
        Names of the features (length # features)

    max_display : int
        How many top features to include in the plot (default is 20, or 7 for interaction plots)

    plot_type : "dot" (default for single output), "bar" (default for multi-output), "violin",
        or "compact_dot".
        What type of summary plot to produce. Note that "compact_dot" is only used for
        SHAP interaction values.

    plot_size : "auto" (default), float, (float, float), or None
        What size to make the plot. By default the size is auto-scaled based on the number of
        features that are being displayed. Passing a single float will cause each row to be that 
        many inches high. Passing a pair of floats will scale the plot by that
        number of inches. If None is passed then the size of the current figure will be left
        unchanged.
    """
    # fig,ax=pl.subplots()

    # support passing an explanation object
    if str(type(shap_values)).endswith("Explanation'>"):
        shap_exp = shap_values
        base_value = shap_exp.base_values
        shap_values = shap_exp.values
        if features is None:
            features = shap_exp.data
        if feature_names is None:
            feature_names = shap_exp.feature_names
        # if out_names is None: # TODO: waiting for slicer support of this
        #     out_names = shap_exp.output_names

    # deprecation warnings
    if auto_size_plot is not None:
        warnings.warn("auto_size_plot=False is deprecated and is now ignored! Use plot_size=None instead.")

    multi_class = False
    if isinstance(shap_values, list):
        multi_class = True
        if plot_type is None:
            plot_type = "bar" # default for multi-output explanations
        assert plot_type == "bar", "Only plot_type = 'bar' is supported for multi-output explanations!"
    else:
        if plot_type is None:
            plot_type = "dot" # default for single output explanations
        assert len(shap_values.shape) != 1, "Summary plots need a matrix of shap_values, not a vector."

    # default color:
    if color is None:
        if plot_type == 'layered_violin':
            color = "coolwarm"
        elif multi_class:
            color = lambda i: colors.red_blue_circle(i/len(shap_values))
        else:
            color = colors.blue_rgb

    idx2cat = None
    # convert from a DataFrame or other types
    if str(type(features)) == "<class 'pandas.core.frame.DataFrame'>":
        if feature_names is None:
            feature_names = features.columns
        # feature index to category flag
        idx2cat = features.dtypes.astype(str).isin(["object", "category"]).tolist()
        features = features.values
    elif isinstance(features, list):
        if feature_names is None:
            feature_names = features
        features = None
    elif (features is not None) and len(features.shape) == 1 and feature_names is None:
        feature_names = features
        features = None

    num_features = (shap_values[0].shape[1] if multi_class else shap_values.shape[1])

    if features is not None:
        shape_msg = "The shape of the shap_values matrix does not match the shape of the " \
                    "provided data matrix."
        if num_features - 1 == features.shape[1]:
            assert False, shape_msg + " Perhaps the extra column in the shap_values matrix is the " \
                          "constant offset? Of so just pass shap_values[:,:-1]."
        else:
            assert num_features == features.shape[1], shape_msg

    if feature_names is None:
        feature_names = np.array([labels['FEATURE'] % str(i) for i in range(num_features)])

    if use_log_scale:
        pl.set_xscale('symlog')

    # plotting SHAP interaction values
    ### TOLTO ###

    if max_display is None:
        max_display = 20

    if sort:
        # order features by the sum of their effect magnitudes
        if custom_order:
            feature_order = custom_order(shap_values)
        else:
            feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
        feature_order = feature_order[-min(max_display, len(feature_order)):]
    else:
        feature_order = np.flip(np.arange(min(max_display, num_features)), 0)

    row_height = 0.4
    if plot_size == "auto":
        pl.gcf().set_size_inches(8, len(feature_order) * row_height + 1.5)
    elif type(plot_size) in (list, tuple):
        pl.gcf().set_size_inches(plot_size[0], plot_size[1])
    elif plot_size is not None:
        pl.gcf().set_size_inches(8, len(feature_order) * plot_size + 1.5)
    pl.axvline(x=0, color="#999999", zorder=-1)

##################################
## GUARDARE QUI PER IL SUMMARY PLOT
##################################
    # if plot_type == "dot": ## UNICO TIPO LASCIATO
    for pos, i in enumerate(feature_order):
        pl.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
        shaps = shap_values[:, i]
        values = None if features is None else features[:, i]
        inds = np.arange(len(shaps))
        np.random.shuffle(inds)
        if values is not None:
            values = values[inds]
        shaps = shaps[inds]
        colored_feature = True
        try:
            if idx2cat is not None and idx2cat[i]: # check categorical feature
                colored_feature = False
                print("toglie colori")
            else:
                values = np.array(values, dtype=np.float64)  # make sure this can be numeric
        except:
            print("?")
            colored_feature = False
        N = len(shaps)
        # hspacing = (np.max(shaps) - np.min(shaps)) / 200
        # curr_bin = []
        nbins = 100
        quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
        inds = np.argsort(quant + np.random.randn(N) * 1e-6)
        layer = 0
        last_bin = -1
        ys = np.zeros(N)
        for ind in inds:
            if quant[ind] != last_bin:
                layer = 0
            ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
            layer += 1
            last_bin = quant[ind]
        ys *= 0.9 * (row_height / np.max(ys + 1))

        if features is not None and colored_feature:
            # trim the color range, but prevent the color range from collapsing
            vmin = np.nanpercentile(values, 5)
            vmax = np.nanpercentile(values, 95)
            if vmin == vmax:
                vmin = np.nanpercentile(values, 1)
                vmax = np.nanpercentile(values, 99)
                if vmin == vmax:
                    vmin = np.min(values)
                    vmax = np.max(values)
            if vmin > vmax: # fixes rare numerical precision issues
                vmin = vmax

            assert features.shape[0] == len(shaps), "Feature and SHAP matrices must have the same number of rows!"

            # plot the nan values in the interaction feature as grey
            nan_mask = np.isnan(values)
            # print ("passa qui ?")
            pl.scatter(shaps[nan_mask], pos + ys[nan_mask], color="#777777", vmin=vmin,
                       vmax=vmax, s=16, alpha=alpha, linewidth=0,
                       zorder=3, rasterized=len(shaps) > 500)

            # plot the non-nan values colored by the trimmed feature value
            cvals = values[np.invert(nan_mask)].astype(np.float64)
            cvals_imp = cvals.copy()
            cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
            cvals[cvals_imp > vmax] = vmax
            cvals[cvals_imp < vmin] = vmin
            pl.scatter(shaps[np.invert(nan_mask)], pos + ys[np.invert(nan_mask)],
                       cmap=cmap, vmin=vmin, vmax=vmax, s=16,
                       c=cvals, alpha=alpha, linewidth=0,
                       zorder=3, rasterized=len(shaps) > 500)
        else:

            pl.scatter(shaps, pos + ys, s=16, alpha=alpha, linewidth=0, zorder=3,
                       color=color if colored_feature else "#777777", rasterized=len(shaps) > 500)

    # TOLTI GLI ALTRI TIPI DI GRAFICI #            

    # draw the color bar
    #######################
    #### IMPOSTARE QUI LA BARRA PER I VALORI FEATURES
    ##################
    if color_bar and features is not None and plot_type != "bar" and \
            (plot_type != "layered_violin" or color in pl.cm.datad):
        
        binary=True # argument della funzione magari :)
        if binary:
            from matplotlib.lines import Line2D
            leg_elems=[
                    Line2D([0],[0], marker='o', color='w', label="Absence", markerfacecolor=cmap.colors[0], markersize=15),
                    Line2D([0],[0], marker='o', color='w', label="Presence", markerfacecolor=cmap.colors[1], markersize=15)
                        ]
            pl.legend(handles=leg_elems, loc='right')
        else:
            import matplotlib.cm as cm
            m = cm.ScalarMappable(cmap=cmap if plot_type != "layered_violin" else pl.get_cmap(color))
            m.set_array([0, 1])
            cb = pl.colorbar(m, ticks=[0, 1], aspect=80)
            cb.set_ticklabels([labels['FEATURE_VALUE_LOW'], labels['FEATURE_VALUE_HIGH']])
            cb.set_label(color_bar_label, size=12, labelpad=0)
            cb.ax.tick_params(labelsize=11, length=0)
            cb.set_alpha(1)
            cb.outline.set_visible(False)
            bbox = cb.ax.get_window_extent().transformed(pl.gcf().dpi_scale_trans.inverted())
            cb.ax.set_aspect((bbox.height - 0.9) * 20)
            cb.draw_all()

    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('none')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    pl.gca().spines['left'].set_visible(False)
    pl.gca().tick_params(color=axis_color, labelcolor=axis_color)

    if features_imgPath:
        ax=pl.gca()
        import os
        from matplotlib import image as matimg
        allyPos=[x.get_position()[1] for x in ax.get_ymajorticklabels()]
        allxTicks=ax.get_xmajorticklabels()
        xlimitInf,xlimitSup=ax.get_xlim()
        larg=(xlimitSup-xlimitInf)*0.2

        # imgNames=[os.getcwd()+features_imgPath+f"/{feature_names[i]}.png" for i in feature_order]        
        imgNames=[os.path.join(features_imgPath,f"{feature_names[i]}.png") for i in feature_order]
        insets=[ax.inset_axes([xlimitInf-larg, i-row_height , larg, row_height*2 ], 
                transform=ax.transData
                # ) for i in allyPos]
                ) for i in range(len(feature_order)) ]
        
        for g,i in zip(insets, imgNames):
            g.axison=False
            g.imshow(  matimg.imread(i)  )
        ax.set_yticks([])

        ax.spines['bottom'].set_visible(False)
        from matplotlib.patches import FancyArrow
        neg=FancyArrow(0, 0, xlimitInf, 0, length_includes_head=True, facecolor="green", edgecolor="green",transform=ax.get_xaxis_transform(), clip_on=False)
        pos=FancyArrow(0, 0, xlimitSup, 0, length_includes_head=True, facecolor="red", edgecolor="red",transform=ax.get_xaxis_transform(), clip_on=False)
        ax.text(xlimitSup/2, -row_height*2, 'Cardiotoxic', size=13, color='red', ha="center", transform=ax.transData)
        ax.text(xlimitInf/2, -row_height*2, 'Safe', size=13, color='green', ha="center", transform=ax.transData)
        # ax.annotate('no-toxic', (xlimitInf/2, 0,), (0, 0), xycoords='data', textcoords='offset points',ha="center", va='bottom', color="green")
        ax.add_patch(neg)
        ax.add_patch(pos)

    else:
        pl.yticks(range(len(feature_order)), [feature_names[i] for i in feature_order], fontsize=13)

    if plot_type != "bar":
        pl.gca().tick_params('y', length=20, width=0.5, which='major')
    pl.gca().tick_params('x', labelsize=11)
    ax.set_ylim(-1, len(feature_order))
    if plot_type == "bar":
        ax.set_xlabel(labels['GLOBAL_VALUE'], fontsize=13)
    else:
        # ax.set_xlabel(labels['VALUE'], fontsize=13)
        ax.set_xlabel("SHAP value", fontsize=13)
    pl.tight_layout()
    if show:
        pl.show()
    else:
        return pl.gcf()