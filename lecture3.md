class: middle, center, title-slide

# Foundations of Data Science

Lecture 3: Visualization

<br><br>
Prof. Gilles Louppe<br>
[g.louppe@uliege.be](mailto:g.louppe@uliege.be)

???

https://badriadhikari.github.io/data-viz-workshop-2021/

---

class: middle

.center.width-55[![](figures/lec2/pairplot_by_species.png)]

Last lecture produced plots like this one. This lecture is about drawing them well.

---

class: middle

.smaller-x.center[
```
   dataset   mean_x   mean_y     sd_x     sd_y         cor
      away 54.26610 47.83472 16.76982 26.93974 -0.06412835
  bullseye 54.26873 47.83082 16.76924 26.93573 -0.06858639
    circle 54.26732 47.83772 16.76001 26.93004 -0.06834336
      dino 54.26327 47.83225 16.76514 26.93540 -0.06447185
      dots 54.26030 47.83983 16.76774 26.93019 -0.06034144
   h_lines 54.26144 47.83025 16.76590 26.93988 -0.06171484
high_lines 54.26881 47.83545 16.76670 26.94000 -0.06850422
slant_down 54.26785 47.83590 16.76676 26.93610 -0.06897974
  slant_up 54.26588 47.83150 16.76885 26.93861 -0.06860921
      star 54.26734 47.83955 16.76896 26.93027 -0.06296110
   v_lines 54.26993 47.83699 16.76996 26.93768 -0.06944557
wide_lines 54.26692 47.83160 16.77000 26.93790 -0.06657523
   x_shape 54.26015 47.83972 16.76996 26.93000 -0.06558334
```
]

.center[.bold[Do not summarize the data without visualizing it.]<br> The Datasaurus dozen: 13 datasets with identical summary statistics.]

.footnote[Credits: [Matejka & Fitzmaurice](https://www.research.autodesk.com/publications/same-stats-different-graphs/), 2017.]

---

class: middle

.center.width-90[![](figures/lec3/datasaurus.png)]

.center[Summary statistics can be misleading. .bold[Always visualize the rawest data]!]

.footnote[Credits: [Matejka & Fitzmaurice](https://www.research.autodesk.com/publications/same-stats-different-graphs/), 2017.]

---

class: middle

## Two kinds of plots

.center.width-100[![](figures/lec3/exploratory-explanatory.png)]

- .bold[Exploratory]: for yourself, quickly and in numbers, to see (Lecture 2).
- .bold[Explanatory]: for others, few and polished, to carry one message.

???

The building blocks are the same, the standards are not. The same plots come back in Lecture 7, on what a model gets wrong.

---

class: middle

# Choosing the right plot

---

class: middle

## Start from the message

.center.width-75[![](figures/lec3/four-questions.png)]

Choose the graph from the question it answers, then the design from that question and the number of continuous variables.

.footnote[Credits: [Doumont](https://www.principiae.be/X0100.php), Trees, maps, and theorems, 2009.]

???

For subsets, either distinguish them within one panel, or juxtapose panels sharing identical scales.

---

class: middle

## What that gives

- .bold[Comparison]: horizontal bars, which must start at zero, or dots along a scale, which need not.
- .bold[Distribution]: every point along a scale; a histogram or a box plot when there are too many.
- .bold[Correlation]: a scatter plot, or an array of them beyond two variables.
- .bold[Evolution]: lines against the independent variable; different units go in different panels sharing a scale.

This is one opinionated but consistent take, not the only one.

---

class: center, middle, black-slide

<iframe width="600" height="400" src="https://www.youtube.com/embed/6lm4wJ1qm0w" frameborder="0" allowfullscreen></iframe>

Jean-Luc Doumont, "Choosing the right graph", 2017.

---

class: middle

# Encoding data with visual cues

---

class: middle

<span style="font-size: 5em;" class="center">14, 37, 75</span>

<br>

.question[In pairs, try to come up with as many representations/encodings of this "data" as possible.]

---

class: middle

## Anatomy of a plot

.center.width-70[![](figures/lec3/anatomy.png)]

.center[data → marks → channels → scales → coordinates → guides]

.footnote[The .bold[grammar of graphics]: Wilkinson, 1999; implemented in ggplot2, Vega-Lite, Altair and seaborn objects.]

???

The rest of the lecture takes these pieces in order.

---

class: middle

## Visual cues

Visual cues are elements of a visualization that encode data. They are expressed as marks and channels:
- A .bold[mark] is a geometric primitive such as points, lines, or areas.
- A .bold[channel] is an attribute of a mark that can be used to encode data, such as position, size, shape, or color.

<br><br>
.center.width-100[![](figures/lec3/marks.png)<br>Marks are geometric primitives.]

.footnote[Credits: T. Munzner, "Visualization Analysis and Design", 2014.]

---

class: middle

.center.width-100[![](figures/lec3/channels.png)<br>Visual channels control the appearance of marks.]

.footnote[Credits: T. Munzner, "Visualization Analysis and Design", 2014.]

---

class: middle

.center.width-100[![](figures/lec3/encoding-examples.png)]

(a) Bar charts encode data using line marks, controlled by a vertical position channel (height) and a horizontal position channel (category).
(b) Scatterplots encode data using point marks, controlled by two position channels (x and y).
(c) A third variable can be encoded using a color channel.
(d) A fourth variable can be encoded using a size channel (area of the point).

---

class: middle

## Overplotting

.center.width-100[![](figures/lec3/overplotting.png)]

Flipper lengths are whole millimetres, body masses multiples of 25 g, so identical readings stack. Transparency, jitter or binning give the density back.

---

class: middle

## Two principles

- .bold[Expressiveness]: show all the data facts, and only the data facts.
- .bold[Effectiveness]: encode them with the channels a reader decodes most accurately.

.footnote[Mackinlay, 1986.]

---

class: middle

## Perceptual hierarchy

Effectiveness is measurable: not all channels are read equally well.

.center.width-50[![](figures/lec3/ladder.png)]

.footnote[Credits: T. Munzner, "Visualization Analysis and Design", 2014.]

---

class: middle

Cleveland and McGill (1984) conducted experiments to evaluate the accuracy of visual channels for encoding quantitative data.

.center.width-80[![](figures/lec3/cleveland.png)]

.footnote[Credits: T. Munzner, "Visualization Analysis and Design", 2014.]

---

class: middle

.center.width-10[![](figures/lec3/color-wheel.png)]

## Colors

Color is a powerful channel for encoding categorical and quantitative data.

The primary representation system is the Hue, Saturation, Value (HSV) model:
- .bold[Hue]: the type of color (e.g., red, green, blue), numerically represented as an angle on the color wheel (0-360 degrees).
- .bold[Saturation]: the intensity or purity of the color (from gray to full color), represented as a percentage (0-100%).
- .bold[Value]: the brightness of the color (from black to full brightness), represented as a percentage (0-100%).

---

class: middle

.center.width-100[![](figures/lec3/hsv.png)]

---

class: middle

A colormap maps data values to colors. Three kinds, for three kinds of data:
- .bold[Sequential]: ordered data, light to dark within a single hue.
- .bold[Diverging]: ordered data with a meaningful midpoint, two hues.
- .bold[Categorical]: unordered groups, distinct colors.

.center.width-100[![](figures/lec3/colormap-types.png)]

---

class: middle

.center[
.width-45[![](figures/lec3/sphx_glr_colormaps_002.png)]

.width-45[![](figures/lec3/sphx_glr_colormaps_004.png)]
.width-45[![](figures/lec3/sphx_glr_colormaps_006.png)]
]

.footnote[Credits: [Choosing colormaps in Matplotlib](https://matplotlib.org/stable/users/explain/colors/colormaps.html).]

---

class: middle

.center.width-10[![](figures/lec3/ishihara-test.png)]

.alert[Color vision deficiency affects approximately .bold[1 in 12 men] and .bold[1 in 200 women] worldwide. .bold[Never let colour alone carry what the reader must not miss]: keep the categories few, pick a colourblind-safe palette, and repeat the distinction with position, shape or a label.] 

---

class: middle

.center.width-75[![](figures/lec3/sphx_glr_colormaps_001.png)]

Perceptually uniform colormaps ensure that equal steps in data are perceived as equal steps in color.

.footnote[Credits: [Choosing colormaps in Matplotlib](https://matplotlib.org/stable/users/explain/colors/colormaps.html).]

???

They should be preferred over non-uniform colormaps (jet, rainbow) that can mislead interpretation.

---

class: middle

Finally, colors can also be used to .bold[draw attention] to specific elements in a visualization, such as highlighting important data points or trends.

.center.width-90[![](figures/lec3/popout.png)]

.center[A single channel pops out at a glance; two channels at once must be searched.]

---

class: middle

## Small multiples

.center.width-100[![](figures/lec3/small-multiples.png)]

Subsets can be told apart within one panel, by colour, or split into panels sharing their scales. Keeping the other points in grey gives each panel its reference.

---

class: middle

.center.width-10[![](figures/lec3/scale.png)]

## Scales and transformations

Sometimes, data spans several orders of magnitude or has a skewed distribution. In such cases, applying a transformation or using a different scale can improve the interpretability of the visualization.
- .bold[Linear scale]: preserves the original data values.
- .bold[Logarithmic scale]: useful for data spanning several orders of magnitude.
- .bold[Quantile scale]: divides data into bins holding equal numbers of records, useful for skewed distributions.

---

class: middle

.center.width-80[![](figures/lec3/scale-linear.png)]

.center[Linear scales can be dominated by large values, dwarfing smaller values and making it hard to see small variations.]

.footnote[Data: `data/countries.csv`.]

---

class: middle

.center.width-80[![](figures/lec3/scale-log.png)]

.center[Showing data on a logarithmic scale can prevent large values from dominating the visualization and reveal patterns among smaller values.]

.center[It also hides absolute differences, cannot show zero or negative values,<br> and is read as linear by an audience that does not expect it.]

.footnote[Data: `data/countries.csv`.]

---

class: middle

## Binning is a choice

.center.width-100[![](figures/lec3/binning.png)]

The same 342 body masses. Too few bins hide the second group, too many show noise as structure. Bin width, like kernel bandwidth, belongs to the plot and not to the data.

---

class: middle

.center.width-10[![](figures/lec3/coordinates.png)]

## Coordinate systems

Before mapping (numerical) data to visual channels, it is important to choose an appropriate coordinate system as it affects the perception and effectiveness of the visualization.
- .bold[Cartesian coordinates]: intuitive and effective for most data types.
- .bold[Polar coordinates]: useful for cyclic data (e.g., time of day, seasons), but can distort perception of lengths.

---

class: middle, black-slide

.center.width-45[![](figures/lec3/spiral.png)]

.center[Climate spiral vs. line chart showing global mean temperature change over time.]

.footnote[Credits: [Open climate data](https://openclimatedata.net/climate-spirals/temperature-line-chart/), adapted from Ed Hawkins' climate spiral.]

---

class: middle

## Guides

.center.width-100[![](figures/lec3/guides.png)]

A legend costs a lookup for every group. Labelling them in place, and annotating the one thing the reader should notice, costs nothing.

---

class: middle

## Show the uncertainty

.center.width-100[![](figures/lec3/uncertainty.png)]

A bar with an error bar shows a mean and a width, and hides everything else. Show the data, then the estimate and its interval.

.alert[An error bar means nothing until the caption says what it is: here the same Adelie mean carries ±459 g, ±37 g or ±73 g, depending on whether the bar is a standard deviation, a standard error or a 95% interval.]

???

From Lecture 7 on, the same holds for what a model predicts.


---

class: middle

# Anti-patterns 

---

class: middle

.center.width-60[![](figures/lec3/pie-chart.png)]

.center[What category is the largest?]

---

class: middle

.center.width-60[![](figures/lec3/bar-chart.png)]

.center[Same data! .bold[Lengths and positions are easier to compare] than angles and areas.]

???

A pie is readable for two or three slices adding up to a whole. Beyond that, it asks the reader to compare angles.

---


class: middle

.center.width-60[![](figures/lec3/3d-pie.png)]

.center[.bold[Do not go for 3D.] It distorts perception and adds unnecessary complexity.]

.footnote[Credits: Claus O. Wilke, [Fundamentals of Data Visualization](https://clauswilke.com/dataviz/), 2019.]

???

Worse than a pie chart? A 3D pie chart.

---

class: middle

.center.width-60[![](figures/lec3/colorado.jpg)]

.center[.bold[Do not exaggerate reality] by truncating axes or using misleading aspect ratios.]

???

The rule follows from the encoding: a bar encodes a value as a length, so cutting its baseline lies. The next slide makes the distinction.

.footnote[Credits: John Muyskens, [Most of Trump's charts skew the data](https://www.washingtonpost.com/graphics/politics/2016-election/trump-charts/), The Washington Post, 2016.]

---

class: middle

.grid[
.kol-1-2[.center.width-65[![](figures/lec3/colorado-fake.png)]]
.kol-1-2[.center.width-65[![](figures/lec3/colorado-real.png)]]
]

.footnote[Credits: John Muyskens, [Most of Trump's charts skew the data](https://www.washingtonpost.com/graphics/politics/2016-election/trump-charts/), The Washington Post, 2016.]

---

class: middle

## Zero belongs under bars

.center.width-95[![](figures/lec3/axis-zero.png)]

A bar encodes a value as a length, so its baseline must be zero. A dot or a line encodes a position, and may show a narrow range: temperature anomalies of a degree do not need an axis starting at zero.

---

class: middle

.center[
.width-45[![](figures/lec3/tufte1.png)] .width-45[![](figures/lec3/tufte2.png)]
]

.center[.bold[Do not lie.] Use visual channels that accurately represent the data.]

.footnote[Credits: David Giard, [Data visualization](https://davidgiard.com/data-visualization-part-3-graphical-integrity).]

---

class: middle

.center[<video controls preload="auto" height="480" width="640">
  <source src="./figures/lec3/dataink.mp4" type="video/mp4">
</video>]

.center[.bold[Maximize the data-ink ratio] by removing unnecessary elements.]

.center[Up to a point: grid lines help read values off a scale,<br> and a memorable chart is sometimes worth more ink (Bateman et al., 2010).]

.footnote[Credits: Joey Cherdarchuk, [Data looks better naked](https://www.darkhorseanalytics.com/blog/data-looks-better-naked).]

???

Antoine de Saint-Exupéry: "Perfection is achieved, not when there is nothing more to add, but when there is nothing left to take away."

---

class: middle

.center.width-10[![](figures/lec3/grade.png)]

## Wrap-up exercise

Let us discuss the following examples. For each of them, identify what is good and what could be improved.

(All plots are taken from MSc theses of previous years. Author names have been removed to protect the innocent.)

---

class: middle

.center.width-100[![](figures/lec3/discussion1.png)]

---

class: middle

.center.width-85[![](figures/lec3/discussion2.png)]

---

class: middle

.center.width-60[![](figures/lec3/discussion3.png)]


---

class: middle

## Key takeaways

- Start from the message: comparison, distribution, correlation or evolution.
- Encode it with the channels read most accurately: position and length first.
- Pick scales, coordinates and colours that do not distort what the data say.
- Avoid the classics: 3D, bars cut off from zero, colour as the only cue.

A good plot answers a question, and does not let the reader misread the answer.

---

class: end-slide, center
count: false

The end.
