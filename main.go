package main

import (
	"image"
	"image/color"
	"sort"

	"github.com/google/uuid"

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
	totalChunks = 20
)

// Heavily inspired by: https://github.com/arthurflor23/text-segmentation/blob/master/src/imgproc/cpp/LineSegmentation.cpp
func segmentLines(origImage gocv.Mat) []gocv.Mat {
	//
	// Steps:
	// Get contours?
	// I think this is just finding a binding box for the area with text. I think we already have this from the original
	// scan.

	// Generate chunks
	allChunks := generateChunks(origImage)
	toOutput := []gocv.Mat{origImage}
	for _, c := range allChunks {
		toOutput = append(toOutput, c.mat)
	}

	// Get the initial lines

	// Generate regions

	// Repair Lines

	// Generate regions (2)

	// Print lines

	// Get regions
	return toOutput
}

type chunkStruct struct {
	index      int
	startPixel int
	chunkWidth int
	mat        gocv.Mat
	histogram  []int
	peaks      []peakStruct
	valleys    []*valleyStruct

	avgHeight, avgWhiteHeight int
	linesCount                int
}

// Returns the average height of something?
func (c *chunkStruct) findPeaksAndValleys(idToValley map[string]*valleyStruct) int {
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
			id:         uuid.New().String(),
			chunkIndex: c.index,
			position:   minPosition,
		}
		c.valleys = append(c.valleys, v)
		idToValley[v.id] = v
	}
	return c.avgHeight
}

func (c *chunkStruct) calculateHistogram() {
	c.histogram = make([]int, c.mat.Cols())
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
	id         string
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
	valleyIDs                      []string
	minRowPosition, maxRowPosition int
	points                         []image.Point
}

func newLine(initialValleyID string) *lineStruct {
	return &lineStruct{
		valleyIDs: []string{initialValleyID},
	}
}

func (l *lineStruct) generateInitialPoints(
	chunkWidth int,
	imageWidth int,
	idToValley map[string]*valleyStruct,
) {
	// todo
}

type regionStruct struct {
	region      gocv.Mat
	top, bottom *lineStruct
	height      int
	rowOffset   int
	covariance  gocv.Mat
	mean        []float32
}

func (r *regionStruct) updateRegion(img *gocv.Mat, something int) {
	// TODO
}

func (r *regionStruct) calculateMean() {
	// TODO
}

func (r *regionStruct) calculateCovariance() {
	// TODO
}

func (r *regionStruct) biVariateGaussianDensity(point image.Point) float64 {
	// TODO
	return 0
}

func generateChunks(origImage gocv.Mat) []chunkStruct {
	width := origImage.Cols() / totalChunks

	chunks := make([]chunkStruct, totalChunks)
	for i, startPixel := 0, 0; i < totalChunks; i++ {
		chunks[i] = chunkStruct{
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
func getInitialLines(chunks []chunkStruct, idToValleys map[string]*valleyStruct) (int, []lineStruct) {
	lines := make([]lineStruct, 0, len(chunks))
	numberOfHeights, valleysMinAbsoluteDistance := 0, 0
	for _, c := range chunks {
		averageHeight := c.findPeaksAndValleys(idToValleys)
		if averageHeight > 0 {
			numberOfHeights++
		}
		valleysMinAbsoluteDistance += averageHeight
	}
	valleysMinAbsoluteDistance /= numberOfHeights

	for i := len(chunks) - 1; i >= 0; i-- {
		for range chunks[i].valleys {
			// tODO: Claim a valley if unclaimed, then make a new line if there's more than one valley??
		}
	}

	return valleysMinAbsoluteDistance, lines
}
