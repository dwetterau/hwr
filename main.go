package main

import (
	"image"
	"image/color"

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
}

// Returns the average height of something?
func (c *chunkStruct) findPeaksAndValleys(idToValley map[string]*valleyStruct) int {
	return 0
}

func (c *chunkStruct) calculateHistogram() {
	// todo
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
	// todo?
	return nil
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
