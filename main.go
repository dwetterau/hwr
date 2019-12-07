package main

import (
	"fmt"
	"image"
	"image/color"
	"math"
	"sort"

	"gocv.io/x/gocv"
)

func main() {
	// open display window
	window := gocv.NewWindow("HRR")
	defer window.Close()

	// prepare image matrix
	origImg := gocv.IMRead("/home/david/Dropbox/Journal/2019-07-25.jpg", gocv.IMReadGrayScale)
	defer origImg.Close()
	if origImg.Empty() {
		panic("didn't load image")
	}
	//images := detectWords(origImg)
	images := segmentLines(origImg)
	i := 0
	for {
		toDraw := images[i%len(images)]
		window.IMShow(toDraw)
		if window.WaitKey(1000) == 27 {
			break
		}
		i++
	}
	for _, i := range images {
		_ = i.Close()
	}
}

func detectWords(origImg gocv.Mat) []gocv.Mat {
	blurred := gocv.NewMat()
	gocv.Blur(origImg, &blurred, image.Point{X: 5, Y: 5})

	thresholded := gocv.NewMat()
	gocv.AdaptiveThreshold(blurred, &thresholded, 255, gocv.AdaptiveThresholdGaussian, gocv.ThresholdBinaryInv, 11, 15)

	dilated := gocv.NewMat()
	dilationKernal := gocv.GetStructuringElement(gocv.MorphRect, image.Point{X: 5, Y: 1})
	dilationKernalTall := gocv.GetStructuringElement(gocv.MorphRect, image.Point{X: 5, Y: 2})
	src, dst := &thresholded, &dilated

	newSrc := gocv.NewMat()
	defer newSrc.Close()

	iterations := 5
	for i := 0; i < iterations; i++ {
		k := dilationKernal
		if i == iterations-1 {
			k = dilationKernalTall
		}
		gocv.Dilate(*src, dst, k)
		newDst := &newSrc
		if i != 0 {
			newDst = src
		}
		src = dst
		dst = newDst
	}
	dilated = *src

	contours := gocv.FindContours(dilated, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	blue := color.RGBA{0, 0, 255, 0}
	rectangles := make([]image.Rectangle, 0, len(contours))
	for _, c := range contours {
		r := gocv.BoundingRect(c)
		s := r.Size()
		if s.X*s.Y > 1000 {
			rectangles = append(rectangles, r)
		}
	}

	final := gocv.NewMat()

	gocv.CopyMakeBorder(origImg, &final, 0, 0, 0, 0, gocv.BorderConstant, blue)
	for _, r := range rectangles {
		gocv.Rectangle(&final, r, blue, 3)
	}

	return []gocv.Mat{origImg, blurred, thresholded, dilated, final}
}

const (
	totalChunks = 50
)

type valleyID int

var valleyIDCounter = valleyID(0)

// Heavily inspired by: https://github.com/arthurflor23/text-segmentation/blob/master/src/imgproc/cpp/LineSegmentation.cpp
func segmentLines(origImage gocv.Mat) []gocv.Mat {
	//
	// Steps:
	// Get contours?
	// I think this is just finding a binding box for the area with text. I think we already have this from the original
	// scan.

	// Generate chunks
	allChunks := generateChunks(origImage)

	// Get the initial lines
	idToValleys := make(map[valleyID]*valleyStruct, 50)
	predictedLineHeight, lines := getInitialLines(allChunks, idToValleys)
	fmt.Println(predictedLineHeight)

	toOutput := []gocv.Mat{origImage}
	toOutput = append(toOutput, origImage.Clone())
	renderLines(toOutput[len(toOutput)-1], lines)

	if len(lines) > 0 {
		// Generate regions

		// Repair Lines

		// Generate regions (2)

		// Render lines

		// Get regions
	}

	renderLines(toOutput[1], lines)
	return toOutput
}

type chunkStruct struct {
	index      int
	startPixel int
	chunkWidth int
	mat        gocv.Mat
	histogram  []int
	peaks      []peakStruct
	valleyIDs  []valleyID

	avgHeight, avgWhiteHeight int
	linesCount                int
}

func (c *chunkStruct) findPeaksAndValleys(idToValley map[valleyID]*valleyStruct) {
	c.calculateHistogram()

	for i := 1; i < len(c.histogram)-1; i++ {
		leftVal := c.histogram[i-1]
		centerVal := c.histogram[i]
		rightVal := c.histogram[i+1]

		// See if we're looking at a peak
		if centerVal >= leftVal && centerVal >= rightVal {
			if len(c.peaks) > 0 {
				last := c.peaks[len(c.peaks)-1]
				if (i-last.position) <= c.avgHeight/2 && centerVal >= last.value {
					c.peaks[len(c.peaks)-1].position = i
					c.peaks[len(c.peaks)-1].value = centerVal
				} else if (i-last.position) <= c.avgHeight/2 && centerVal < last.value {
					// Pretend it's not a peak
				} else {
					c.peaks = append(c.peaks, peakStruct{position: i, value: centerVal})
				}
			} else {
				c.peaks = append(c.peaks, peakStruct{position: i, value: centerVal})
			}
		}
	}

	peaksAverageValues := 0
	for _, p := range c.peaks {
		peaksAverageValues += p.value
	}
	if len(c.peaks) > 0 {
		peaksAverageValues /= len(c.peaks)
	}
	newPeaks := make([]peakStruct, 0, len(c.peaks))
	for _, p := range c.peaks {
		if p.value >= peaksAverageValues/4 {
			newPeaks = append(newPeaks, p)
		}
	}
	c.linesCount = len(newPeaks)
	c.peaks = newPeaks

	sort.Slice(c.peaks, func(i, j int) bool {
		return c.peaks[i].position < c.peaks[j].position
	})

	for i := 1; i < len(c.peaks); i++ {
		minPosition := (c.peaks[i-1].position + c.peaks[i].position) / 2
		minValue := c.histogram[minPosition]

		for j := c.peaks[i-1].position + c.avgHeight/2; j < c.peaks[i].position-c.avgHeight-30; j++ {
			valleyBlackCount := 0
			for l := 0; l < c.mat.Cols(); l++ {
				if c.mat.GetUCharAt(j, l) == 0 {
					valleyBlackCount++
				}
			}

			if minValue != 0 && valleyBlackCount <= minValue {
				minValue = valleyBlackCount
				minPosition = j
			}
		}

		v := &valleyStruct{
			id:         valleyIDCounter,
			chunkIndex: c.index,
			position:   minPosition,
		}
		valleyIDCounter++
		c.valleyIDs = append(c.valleyIDs, v.id)
		idToValley[v.id] = v
	}
}

func (c *chunkStruct) calculateHistogram() {
	c.histogram = make([]int, c.mat.Rows())
	c.linesCount, c.avgHeight, c.avgWhiteHeight = 0, 0, 0

	blackCount, currentHeight, currentWhiteCount, whiteLinesCount := 0, 0, 0, 0
	var whiteSpaces []int

	for i := 0; i < c.mat.Rows(); i++ {
		blackCount = 0
		for j := 0; j < c.mat.Cols(); j++ {
			if c.mat.GetUCharAt(i, j) == 0 {
				blackCount++
				c.histogram[i]++
			}
		}
		if blackCount > 0 {
			currentHeight++
			if currentWhiteCount > 0 {
				whiteSpaces = append(whiteSpaces, currentWhiteCount)
				currentWhiteCount = 0
			}
		} else {
			currentWhiteCount++
			if currentHeight > 0 {
				c.linesCount++
				c.avgHeight += currentHeight
				currentHeight = 0
			}
		}
	}

	sort.Slice(whiteSpaces, func(i, j int) bool {
		return whiteSpaces[i] < whiteSpaces[j]
	})
	for _, space := range whiteSpaces {
		// Why 4? Idk
		if space > 4*c.avgHeight {
			break
		}
		c.avgWhiteHeight += space
		whiteLinesCount++
	}
	if whiteLinesCount > 0 {
		c.avgWhiteHeight /= whiteLinesCount
	}
	if c.linesCount > 0 {
		c.avgHeight /= c.linesCount
	}

	// Some basic average height logic?
	c.avgHeight += int(float64(c.avgHeight) / 2.0)
	if c.avgHeight < 30 {
		c.avgHeight = 30
	}
}

type valleyStruct struct {
	id         valleyID
	chunkIndex int
	position   int
	used       bool
	line       *lineStruct
}

func (v *valleyStruct) isSame(v2 *valleyStruct) bool {
	return v.id == v2.id
}

type peakStruct struct {
	position int
	value    int
}

type lineStruct struct {
	above, below                   *regionStruct
	valleyIDs                      []valleyID
	minRowPosition, maxRowPosition int
	points                         []image.Point
}

func (l *lineStruct) generateInitialPoints(
	chunkWidth int,
	imageWidth int,
	idToValley map[valleyID]*valleyStruct,
) {
	c, previousRow := 0, 0
	sort.Slice(l.valleyIDs, func(i, j int) bool {
		return l.valleyIDs[i] < l.valleyIDs[j]
	})

	firstV := idToValley[l.valleyIDs[0]]
	if firstV.chunkIndex > 0 {
		previousRow = firstV.position
		l.maxRowPosition = previousRow
		l.minRowPosition = previousRow

		for j := 0; j < firstV.chunkIndex*chunkWidth; j++ {
			if c == j {
				l.points = append(l.points, image.Point{X: previousRow, Y: j})
			}
			c++
		}
	}

	for _, vid := range l.valleyIDs {
		chunkIndex := idToValley[vid].chunkIndex
		chunkRow := idToValley[vid].position
		chunkStartColumn := chunkIndex * chunkWidth

		for j := chunkStartColumn; j < chunkStartColumn+chunkWidth; j++ {
			if chunkRow < l.minRowPosition {
				l.minRowPosition = chunkRow
			}
			if chunkRow > l.maxRowPosition {
				l.maxRowPosition = chunkRow
			}
			if c == j {
				l.points = append(l.points, image.Point{X: chunkRow, Y: j})
			}
			c++
		}
		if previousRow != chunkRow {
			previousRow = chunkRow
			if chunkRow < l.minRowPosition {
				l.minRowPosition = chunkRow
			}
			if chunkRow > l.maxRowPosition {
				l.maxRowPosition = chunkRow
			}
		}
	}

	lastValley := idToValley[l.valleyIDs[len(l.valleyIDs)-1]]
	if totalChunks-1 > lastValley.chunkIndex {
		chunkIndex := lastValley.chunkIndex
		chunkRow := lastValley.position

		for j := chunkIndex*chunkWidth + chunkWidth; j < imageWidth; j++ {
			if c == j {
				l.points = append(l.points, image.Point{X: chunkRow, Y: j})
			}
			c++
		}
	}
}

type regionStruct struct {
	regionIndex int
	region      gocv.Mat
	top, bottom *lineStruct
	height      int
	rowOffset   int
	covariance  gocv.Mat
	mean        []float32
}

// Returns true if the region is all zeros, false otherwise
func (r *regionStruct) updateRegion(img gocv.Mat, regionIndex int) bool {
	r.regionIndex = regionIndex
	minRegionRow := 0
	if r.top != nil {
		minRegionRow = r.top.minRowPosition
	}
	r.rowOffset = minRegionRow
	maxRegionRow := img.Rows()
	if r.bottom != nil {
		maxRegionRow = r.bottom.maxRowPosition
	}

	start := minRegionRow
	if maxRegionRow < start {
		start = maxRegionRow
	}
	end := maxRegionRow
	if minRegionRow > end {
		end = minRegionRow
	}

	regionTotalRows := end - start
	region := gocv.NewMatWithSize(regionTotalRows, img.Cols(), gocv.MatTypeCV8U)
	nonzero := 0
	for c := 0; c < img.Cols(); c++ {
		start := 0
		if r.top != nil {
			start = r.top.points[c].X
		}
		end := img.Rows() - 1
		if r.bottom != nil {
			end = r.bottom.points[c].Y
		}
		for r := 0; r < regionTotalRows; r++ {
			origRow := r + minRegionRow
			val := uint8(255)
			if origRow >= start && origRow < end {
				val = img.GetUCharAt(origRow, c)
			}
			if val != 0 {
				nonzero++
			}
			region.SetUCharAt(r, c, val)
		}
	}
	r.calculateMean()
	r.calculateCovariance()
	return nonzero == region.Cols()*region.Rows()
}

func (r *regionStruct) calculateMean() {
	mean := make([]float32, 2)
	n := 0
	for i := 0; i < r.region.Rows(); i++ {
		for j := 0; j < r.region.Cols(); j++ {
			if r.region.GetUCharAt(i, j) == 255 {
				continue
			}

			if n == 0 {
				mean[0] = float32(i + r.rowOffset)
				mean[1] = float32(j)
			} else {
				fn := float32(n)
				f := (fn - 1.0) / fn
				mean[0] = f*mean[0] + (1-f)*float32(i+r.rowOffset)
				mean[1] = f*mean[1] + (1-f)*float32(j)
			}
			n++
		}
	}
}

func (r *regionStruct) calculateCovariance() {
	r.covariance = gocv.NewMatWithSize(2, 2, gocv.MatTypeCV32F)
	n := 0
	var sumISquared, sumJSquared, sumIJ float32 = 0.0, 0.0, 0.0
	for i := 0; i < r.region.Rows(); i++ {
		for j := 0; j < r.region.Cols(); j++ {
			if r.region.GetUCharAt(i, j) == 255 {
				continue
			}

			newI := float32(i+r.rowOffset) - r.mean[0]
			newJ := float32(j) - r.mean[1]

			sumISquared += newI * newI
			sumIJ += newI * newJ
			sumJSquared += newJ * newJ
			n++
		}
	}

	if n == 0 {
		return
	}
	nf := float32(n)
	r.covariance.SetFloatAt(0, 0, sumISquared/nf)
	r.covariance.SetFloatAt(0, 1, sumIJ/nf)
	r.covariance.SetFloatAt(1, 0, sumIJ/nf)
	r.covariance.SetFloatAt(1, 1, sumJSquared/nf)
}

func (r *regionStruct) biVariateGaussianDensity(point image.Point) float64 {
	// TODO
	return 0
}

func generateChunks(origImage gocv.Mat) []*chunkStruct {
	width := origImage.Cols() / totalChunks

	chunks := make([]*chunkStruct, totalChunks)
	for i, startPixel := 0, 0; i < totalChunks; i++ {
		chunks[i] = &chunkStruct{
			index:      i,
			startPixel: startPixel,
			chunkWidth: width,
			mat:        origImage.Region(image.Rect(startPixel, 0, startPixel+width, origImage.Rows())),
		}
		startPixel += width
	}
	return chunks
}

// Returns predicted line height and the initial lines
func getInitialLines(chunks []*chunkStruct, idToValleys map[valleyID]*valleyStruct) (int, []*lineStruct) {
	lines := make([]*lineStruct, 0, len(chunks))
	numberOfHeights, valleysMinAbsoluteDistance := 0, 0
	for _, c := range chunks {
		c.findPeaksAndValleys(idToValleys)
		if c.avgHeight > 0 {
			numberOfHeights++
		}
		valleysMinAbsoluteDistance += c.avgHeight
	}
	valleysMinAbsoluteDistance /= numberOfHeights

	for i := len(chunks) - 1; i >= 0; i-- {
		for _, vid := range chunks[i].valleyIDs {
			v := idToValleys[vid]
			if v.used {
				continue
			}
			v.used = true
			l := &lineStruct{
				valleyIDs: []valleyID{vid},
			}
			connectValleys(chunks, idToValleys, i-1, vid, l, valleysMinAbsoluteDistance)
			l.generateInitialPoints(chunks[i].chunkWidth, chunks[i].mat.Cols(), idToValleys)

			if len(l.valleyIDs) > 0 {
				lines = append(lines, l)
			}
		}
	}

	return valleysMinAbsoluteDistance, lines
}

func connectValleys(
	chunks []*chunkStruct,
	idToValleys map[valleyID]*valleyStruct,
	i int,
	current valleyID,
	l *lineStruct,
	valleysMinAbsoluteDistance int,
) {
	if i <= 0 || len(chunks[i].valleyIDs) == 0 {
		return
	}

	currentValley := idToValleys[current]
	connectedTo := -1
	minDistance := math.MaxInt64
	for j := 0; j < len(chunks[i].valleyIDs); j++ {
		vid := chunks[i].valleyIDs[j]
		v := idToValleys[vid]
		if v.used {
			continue
		}

		dist := v.position - currentValley.position
		if dist < 0 {
			dist = -dist
		}
		if minDistance > dist && dist <= valleysMinAbsoluteDistance {
			minDistance = dist
			connectedTo = j
		}
	}
	if connectedTo == -1 {
		return
	}

	newV := idToValleys[chunks[i].valleyIDs[connectedTo]]
	l.valleyIDs = append(l.valleyIDs, newV.id)
	newV.used = true
	connectValleys(chunks, idToValleys, i-1, newV.id, l, valleysMinAbsoluteDistance)
}

func renderLines(mat gocv.Mat, lines []*lineStruct) {
	for _, l := range lines {
		lastRow := -1
		for _, p := range l.points {
			mat.SetUCharAt(p.X, p.Y, 255)
			if lastRow != -1 && p.X != lastRow {
				min := p.X
				if lastRow < min {
					min = lastRow
				}
				max := p.X
				if lastRow > max {
					max = lastRow
				}
				for i := min; i < max; i++ {
					mat.SetUCharAt(i, p.Y, 255)
				}
			}
			lastRow = p.X
		}
	}
}

func generateRegions(
	origImage gocv.Mat,
	lines []*lineStruct,
	predictedLineHeight int,
) ([]*regionStruct, int) {
	sort.Slice(lines, func(i, j int) bool {
		return lines[i].minRowPosition < lines[j].minRowPosition
	})

	lineRegions := make([]*regionStruct, 0, len(lines))
	r := &regionStruct{bottom: lines[0]}
	r.updateRegion(origImage, 0)

	lines[0].above = lineRegions[0]
	lineRegions = append(lineRegions, r)

	regionMaxHeight := int(float64(predictedLineHeight) * 2.5)
	averageLineHeight := 0
	if r.height < regionMaxHeight {
		averageLineHeight += r.height
	}

	for i := range lines {
		topLine := lines[i]
		var bottomLine *lineStruct
		if i < len(lines)-1 {
			bottomLine = lines[i+1]
		}

		r := &regionStruct{top: topLine, bottom: bottomLine}
		res := r.updateRegion(origImage, i)
		topLine.below = r
		if bottomLine != nil {
			bottomLine.above = r
		}
		if !res {
			lineRegions = append(lineRegions, r)
			if r.height < regionMaxHeight {
				averageLineHeight += r.height
			}
		}
	}
	if len(lineRegions) > 0 {
		averageLineHeight /= len(lineRegions)
	}
	return lineRegions, averageLineHeight
}
