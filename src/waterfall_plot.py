import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib
from shap.plots._labels import labels
from shap.utils import safe_isinstance, format_value
from shap.plots import colors

def waterfall(shap_values, max_display=10, show=True, plot_size="auto", features_imgPath=None, colorNeg=colors.blue_rgb, colorPos=colors.red_rgb):
    """ Plots an explantion of a single prediction as a waterfall plot.

    The SHAP value of a feature represents the impact of the evidence provided by that feature on the model's
    output. The waterfall plot is designed to visually display how the SHAP values (evidence) of each feature
    move the model output from our prior expectation under the background data distribution, to the final model
    prediction given the evidence of all the features. Features are sorted by the magnitude of their SHAP values
    with the smallest magnitude features grouped together at the bottom of the plot when the number of features
    in the models exceeds the max_display parameter.

    Parameters
    ----------
    shap_values : Explanation
        A one-dimensional Explanation object that contains the feature values and SHAP values to plot.

    max_display : str
        The maximum number of features to plot.

    show : bool
        Whether matplotlib.pyplot.show() is called before returning. Setting this to False allows the plot
        to be customized further after it has been created.
    """

    # Turn off interactive plot
    if show is False:
        plt.ioff()

    base_values = shap_values.base_values
    features = shap_values.display_data if shap_values.display_data is not None else shap_values.data
    feature_names = shap_values.feature_names
    lower_bounds = getattr(shap_values, "lower_bounds", None)
    upper_bounds = getattr(shap_values, "upper_bounds", None)
    values = shap_values.values

    # make sure we only have a single output to explain
    if (type(base_values) == np.ndarray and len(base_values) > 0) or type(base_values) == list:
        raise Exception("waterfall_plot requires a scalar base_values of the model output as the first "
                        "parameter, but you have passed an array as the first parameter! "
                        "Try shap.waterfall_plot(explainer.base_values[0], values[0], X[0]) or "
                        "for multi-output models try "
                        "shap.waterfall_plot(explainer.base_values[0], values[0][0], X[0]).")

    # make sure we only have a single explanation to plot
    if len(values.shape) == 2:
        raise Exception(
            "The waterfall_plot can currently only plot a single explanation but a matrix of explanations was passed!")

    # unwrap pandas series
    if safe_isinstance(features, "pandas.core.series.Series"):
        if feature_names is None:
            feature_names = list(features.index)
        features = features.values

    # fallback feature names
    if feature_names is None:
        feature_names = np.array([labels['FEATURE'] % str(i) for i in range(len(values))])

    # init variables we use for tracking the plot locations
    num_features = min(max_display, len(values))
    row_height = 0.5
    rng = range(num_features - 1, -1, -1)
    order = np.argsort(-np.abs(values))
    pos_lefts = []
    pos_inds = []
    pos_widths = []
    pos_low = []
    pos_high = []
    neg_lefts = []
    neg_inds = []
    neg_widths = []
    neg_low = []
    neg_high = []
    loc = base_values + values.sum()
    yticklabels = ["" for i in range(num_features + 1)]

    # size the plot based on how many features we are plotting
    ### plt.gcf().set_size_inches(8, num_features * row_height + 1.5)
    if plot_size == "auto":
        plt.gcf().set_size_inches(8, num_features * row_height + 1.5)
    elif type(plot_size) in (list, tuple):
        plt.gcf().set_size_inches(plot_size[0], plot_size[1])
    elif plot_size is not None:
        plt.gcf().set_size_inches(8, num_features * plot_size + 1.5)

    # see how many individual (vs. grouped at the end) features we are plotting
    if num_features == len(values):
        num_individual = num_features
    else:
        num_individual = num_features - 1

    # compute the locations of the individual features and plot the dashed connecting lines
    for i in range(num_individual):
        sval = values[order[i]]
        loc -= sval
        if sval >= 0:
            pos_inds.append(rng[i])
            pos_widths.append(sval)
            if lower_bounds is not None:
                pos_low.append(lower_bounds[order[i]])
                pos_high.append(upper_bounds[order[i]])
            pos_lefts.append(loc)
        else:
            neg_inds.append(rng[i])
            neg_widths.append(sval)
            if lower_bounds is not None:
                neg_low.append(lower_bounds[order[i]])
                neg_high.append(upper_bounds[order[i]])
            neg_lefts.append(loc)
        if num_individual != num_features or i + 4 < num_individual:
            plt.plot([loc, loc], [rng[i] - 1 - 0.4, rng[i] + 0.4],
                     color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
        if features is None:
            yticklabels[rng[i]] = feature_names[order[i]]
        else:
            if np.issubdtype(type(features[order[i]]), np.number):
                yticklabels[rng[i]] = format_value(float(features[order[i]]), "%0.03f") + " = " + feature_names[order[i]]
            else:
                yticklabels[rng[i]] = features[order[i]] + " = " + feature_names[order[i]]

    # add a last grouped feature to represent the impact of all the features we didn't show
    if num_features < len(values):
        yticklabels[0] = "%d other features" % (len(values) - num_features + 1)
        remaining_impact = base_values - loc
        if remaining_impact < 0:
            pos_inds.append(0)
            pos_widths.append(-remaining_impact)
            pos_lefts.append(loc + remaining_impact)
            c = colorPos
        else:
            neg_inds.append(0)
            neg_widths.append(-remaining_impact)
            neg_lefts.append(loc + remaining_impact)
            c = colorNeg

    points = pos_lefts + list(np.array(pos_lefts) + np.array(pos_widths)) + neg_lefts + \
        list(np.array(neg_lefts) + np.array(neg_widths))
    dataw = np.max(points) - np.min(points)

    # draw invisible bars just for sizing the axes
    label_padding = np.array([0.1*dataw if w < 1 else 0 for w in pos_widths])
    plt.barh(pos_inds, np.array(pos_widths) + label_padding + 0.02*dataw,
             left=np.array(pos_lefts) - 0.01*dataw, color=colorPos, alpha=0)
    label_padding = np.array([-0.1*dataw if -w < 1 else 0 for w in neg_widths])
    plt.barh(neg_inds, np.array(neg_widths) + label_padding - 0.02*dataw,
             left=np.array(neg_lefts) + 0.01*dataw, color=colorNeg, alpha=0)

    # define variable we need for plotting the arrows
    head_length = 0.08
    bar_width = 0.8
    xlen = plt.xlim()[1] - plt.xlim()[0]
    fig = plt.gcf()
    ax = plt.gca()
    xticks = ax.get_xticks()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    bbox_to_xscale = xlen/width
    hl_scaled = bbox_to_xscale * head_length
    renderer = fig.canvas.get_renderer()

    # draw the positive arrows
    for i in range(len(pos_inds)):
        dist = pos_widths[i]
        arrow_obj = plt.arrow(
            pos_lefts[i], pos_inds[i], max(dist-hl_scaled, 0.000001), 0,
            head_length=min(dist, hl_scaled),
            color=colorPos, width=bar_width,
            head_width=bar_width
        )

        if pos_low is not None and i < len(pos_low):
            plt.errorbar(
                pos_lefts[i] + pos_widths[i], pos_inds[i],
                xerr=np.array([[pos_widths[i] - pos_low[i]], [pos_high[i] - pos_widths[i]]]),
                ecolor=colors.light_red_rgb
            )

        txt_obj = plt.text(
            pos_lefts[i] + 0.5*dist, pos_inds[i], format_value(pos_widths[i], '%+0.02f'),
            horizontalalignment='center', verticalalignment='center', color="white",
            fontsize=12
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)

        # if the text overflows the arrow then draw it after the arrow
        if text_bbox.width > arrow_bbox.width:
            txt_obj.remove()

            txt_obj = plt.text(
                pos_lefts[i] + (5/72)*bbox_to_xscale + dist, pos_inds[i], format_value(pos_widths[i], '%+0.02f'),
                horizontalalignment='left', verticalalignment='center', color=colorPos,
                fontsize=12
            )

    # draw the negative arrows
    for i in range(len(neg_inds)):
        dist = neg_widths[i]

        arrow_obj = plt.arrow(
            neg_lefts[i], neg_inds[i], -max(-dist-hl_scaled, 0.000001), 0,
            head_length=min(-dist, hl_scaled),
            color=colorNeg, width=bar_width,
            head_width=bar_width
        )

        if neg_low is not None and i < len(neg_low):
            plt.errorbar(
                neg_lefts[i] + neg_widths[i], neg_inds[i],
                xerr=np.array([[neg_widths[i] - neg_low[i]], [neg_high[i] - neg_widths[i]]]),
                ecolor=colors.light_blue_rgb
            )

        txt_obj = plt.text(
            neg_lefts[i] + 0.5*dist, neg_inds[i], format_value(neg_widths[i], '%+0.02f'),
            horizontalalignment='center', verticalalignment='center', color="white",
            fontsize=12
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)

        # if the text overflows the arrow then draw it after the arrow
        if text_bbox.width > arrow_bbox.width:
            txt_obj.remove()

            txt_obj = plt.text(
                neg_lefts[i] - (5/72)*bbox_to_xscale + dist, neg_inds[i], format_value(neg_widths[i], '%+0.02f'),
                horizontalalignment='right', verticalalignment='center', color=colorNeg,
                fontsize=12
            )

    # draw the y-ticks twice, once in gray and then again with just the feature names in black
    # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    ytick_pos = list(range(num_features)) + list(np.arange(num_features)+1e-8)
    
    if features_imgPath:
        plt.yticks([ytick_pos[0]], [yticklabels[0]], fontsize=13)
    else:
        plt.yticks(ytick_pos, yticklabels[:-1] + [l.split('=')[-1] for l in yticklabels[:-1]], fontsize=13)

    # put horizontal lines for each feature row
    for i in range(num_features):
        plt.axhline(i, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)

    # mark the prior expected value and the model prediction
    plt.axvline(base_values, 0, 1/num_features, color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
    fx = base_values + values.sum()
    plt.axvline(fx, 0, 1, color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)

    # clean up the main axis
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('none')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    ax.tick_params(labelsize=13)
    #plt.xlabel("\nModel output", fontsize=12)

    # draw the E[f(X)] tick mark
    xmin, xmax = ax.get_xlim()
    ax2 = ax.twiny()
    ax2.set_xlim(xmin, xmax)
    ax2.set_xticks([base_values, base_values+1e-8])  # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    ax2.set_xticklabels(["\n$E[f(X)]$", "\n$ = "+format_value(base_values, "%0.03f")+"$"], fontsize=12, ha="left")
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    # draw the f(x) tick mark
    ax3 = ax2.twiny()
    ax3.set_xlim(xmin, xmax)
    # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    ax3.set_xticks([base_values + values.sum(), base_values + values.sum() + 1e-8])
    ax3.set_xticklabels(["$f(x)$", "$ = "+format_value(fx, "%0.03f")+"$"], fontsize=12, ha="left")
    tick_labels = ax3.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(tick_labels[0].get_transform(
    ) + matplotlib.transforms.ScaledTranslation(-10/72., 0, fig.dpi_scale_trans))
    tick_labels[1].set_transform(tick_labels[1].get_transform(
    ) + matplotlib.transforms.ScaledTranslation(12/72., 0, fig.dpi_scale_trans))
    tick_labels[1].set_color("#999999")
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['left'].set_visible(False)

    # adjust the position of the E[f(X)] = x.xx label
    tick_labels = ax2.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(tick_labels[0].get_transform(
    ) + matplotlib.transforms.ScaledTranslation(-20/72., 0, fig.dpi_scale_trans))
    tick_labels[1].set_transform(tick_labels[1].get_transform(
    ) + matplotlib.transforms.ScaledTranslation(22/72., -1/72., fig.dpi_scale_trans))

    tick_labels[1].set_color("#999999")
    

    if features_imgPath:
        import os
        from matplotlib import image as matimg
        #allxTicks=ax.get_xmajorticklabels()
        #xlimitInf=allxTicks[0].get_position()[0]
        xlimitInf,xlimitSup=ax.get_xlim()
        larg=(xlimitSup-xlimitInf)*0.2

        imgNames=[os.path.join(features_imgPath, f"{l.split('=')[-1].strip()}.png")  for l in yticklabels[1:num_individual+1]]
        insets=[ax.inset_axes([xlimitInf-larg, i-row_height , larg, row_height*2 ], 
                transform=ax.transData
                ) for i in ytick_pos[1:num_individual+1]]
        binX=[int(l.split('=')[0].strip()) for l in yticklabels[1:num_individual+1] ]
        
        for g,i,b,yPos in zip(insets, imgNames, binX, ytick_pos[1:num_individual+1] ):
            ax.text(xlimitInf-larg*2, yPos, "Presence" if b==1 else "Absence", horizontalalignment='left', verticalalignment='bottom', wrap=True)
            g.axison=False
            g.imshow(  matimg.imread(i), aspect="equal"  )

    else:
        # color the y tick labels that have the feature values as gray
        # (these fall behind the black ones with just the feature name)
        tick_labels = ax.yaxis.get_majorticklabels()
        for i in range(num_features):
            tick_labels[i].set_color("#999999")
    

    if show:
        plt.show(plt.gcf())
    else:
        return plt.gcf()
    


color_map = (
        # (0.0, 0.0,1.0,0.5),
        (0.0,1.0,0.0,0.5),
        (1, 1, 1, 0.5),
        (1.0, 0.0, 0.0, 0.5),
        )

def MapSHAP(smi, array, shap_values, FP):
    from compchemkit.machine_learning import feature_importance as feim
    from rdkit import Chem

    vis_cpd_mol_obj = Chem.MolFromSmiles(smi)
    vis_cpd_present_shap = shap_values.values * array.squeeze()
    atom_w = feim.shap2atomweight(vis_cpd_mol_obj, FP, vis_cpd_present_shap)
    svg_image = feim.rdkit_gaussplot(vis_cpd_mol_obj, atom_w, n_contourLines=7, 
                                     color_tuple=color_map)
    
    fig = feim.show_png(svg_image.GetDrawingText())
    # return CropImg(fig)
    return fig