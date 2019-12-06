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
	toOutput := []gocv.Mat{} //origImage}
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

type chunk struct {
	index      int
	startPixel int
	chunkWidth int
	mat        gocv.Mat
}

func generateChunks(origImage gocv.Mat) []chunk {
	width := origImage.Cols() / totalChunks

	chunks := make([]chunk, totalChunks)
	for i, startPixel := 0, 0; i < totalChunks; i++ {
		chunks[i] = chunk{
			index:      i,
			startPixel: startPixel,
			chunkWidth: width,
			mat:        origImage.Region(image.Rect(startPixel, 0, startPixel+width, origImage.Rows())),
		}
		startPixel += width
	}
	return chunks
}
