/* Edge Impulse ingestion SDK
 * Copyright (c) 2022 EdgeImpulse Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License forM the specific language governing permissions and
 * limitations under the License.
 *
 */

// If your target is limited in memory remove this macro to save 10K RAM
#define EIDSP_QUANTIZE_FILTERBANK   0

/**
 * Define the number of slices per model window. E.g. a model window of 1000 ms
 * with slices per model window set to 4. Results in a slice size of 250 ms.
 * For more info: https://docs.edgeimpulse.com/docs/continuous-audio-sampling
 */
#define EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW 1

/*
 ** NOTE: If you run into TFLite arena allocation issue.
 **
 ** This may be due to may dynamic memory fragmentation.
 ** Try defining "-DEI_CLASSIFIER_ALLOCATION_STATIC" in boards.local.txt (create
 ** if it doesn't exist) and copy this file to
 ** `<ARDUINO_CORE_INSTALL_PATH>/arduino/hardware/<mbed_core>/<core_version>/`.
 **
 ** See
 ** (https://support.arduino.cc/hc/en-us/articles/360012076960-Where-are-the-installed-cores-located-)
 ** to find where Arduino installs cores on your machine.
 **
 ** If the problem persists then there's not enough memory for this model and application.
 */

/* Includes ---------------------------------------------------------------- */
#include <PDM.h>
#include <DSP_test_inferencing.h>

/** Audio buffers, pointers and selectors */
typedef struct {
    signed short *buffers[2];
    unsigned char buf_select;
    unsigned char buf_ready;
    unsigned int buf_count;
    unsigned int n_samples;
} inference_t;

static inference_t inference;
static bool record_ready = false;
static signed short *sampleBuffer;
static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal
static int print_results = -(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW);

//size needed for the mfcc buffer, seems to report just 1xnum filters
matrix_size_t mfe_buffer_size = speechpy::feature::calculate_mfe_buffer_size(
                EI_CLASSIFIER_SLICE_SIZE,
                EI_CLASSIFIER_FREQUENCY,
                ei_dsp_config_4.frame_length,
                ei_dsp_config_4.frame_stride,
                ei_dsp_config_4.num_filters,
                ei_dsp_config_4.implementation_version);

ei::matrix_t outputMatrix(1,EI_CLASSIFIER_NN_INPUT_FRAME_SIZE);



/* NEOPIXEL STUFF -----------------------------------------------------------------*/
#include <Adafruit_NeoPixel.h>

#define NUMPIXELS 41 // limited because the neopixel writing is rather slow, needs to be threaded
//needs to be divisable by 2 with remainder 1
#define PIN        9 // for some reason the pin mapping does not exaxtly match that printed 
Adafruit_NeoPixel pixels(NUMPIXELS, PIN, NEO_RGBW + NEO_KHZ800);


/* GRAPH PLOTTING (no Arduino IDE and cutecom support, only puttY confimred so far)--------------------------------------*/
bool printGraph = false;
int nonPrintCycles = 0;
int printEvery = 30;
int graphMaxLength = 100.0;

#define LINE_CASCADING 2
#define SYMMETRIC_CASCADING 3
#define SINGLE_CEPTRUM 1
int ceptrumToShow = 0;

int outputMode = SYMMETRIC_CASCADING;

struct RGBColour{
  uint8_t r;
  uint8_t g;
  uint8_t b;
};

RGBColour pixelArray[NUMPIXELS];

RGBColour pixelArrayOld[NUMPIXELS];

float alphaLowPass = 0.6;

#define SIGMA_FINAL_GAUSSIAN 0.2



float kernelCache[2*NUMPIXELS];


/**
 * @brief      Arduino setup function
 */
void setup()
{

    // put your setup code here, to run once:
  Serial.begin(115200);
  //initialize and test neoPixel

  if(NUMPIXELS % 2 != 1 && outputMode == SYMMETRIC_CASCADING){
    //turn strip red
    for(int i=NUMPIXELS; i>0; i--){
      pixels.setPixelColor(i,255,0,0);
    }
    pixels.show();
    //wait for serial
    while (!Serial);
    //do not run the rest of the code
    while(1){
      Serial.println("ERROR: NUMPIXELS must be an ueneven number");
      delay(2000);
    }
  }
  pixels.begin();
  pixels.setBrightness(255);
  pixels.setPixelColor(1, 255, 255, 255);
  for(int j=0; j<NUMPIXELS; j++){
    for(int i=NUMPIXELS; i>0; i--){
      pixels.setPixelColor(i,pixels.getPixelColor(i-1));
    }
    pixels.show();
    delay(50);
  }

  //calculate kernelCache
  computeKernelCache(kernelCache, NUMPIXELS, SIGMA_FINAL_GAUSSIAN);
  
  Serial.println("Edge Impulse Inferencing Demo");

  // summary of inferencing settings (from model_metadata.h)
  ei_printf("Inferencing settings:\n");
  ei_printf("\tInterval: %.2f ms.\n", (float)EI_CLASSIFIER_INTERVAL_MS);
  ei_printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
  ei_printf("\tSample length: %d ms.\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT / 16);
  ei_printf("\tNo. of classes: %d\n", sizeof(ei_classifier_inferencing_categories) /
                                          sizeof(ei_classifier_inferencing_categories[0]));
  ei_printf("\tNumber of NN_Input: %d\n", EI_CLASSIFIER_NN_INPUT_FRAME_SIZE);
  ei_printf("\tIdeal output size: %dx%d\n", mfe_buffer_size.cols, mfe_buffer_size.rows);

  run_classifier_init();
  if (microphone_inference_start(EI_CLASSIFIER_SLICE_SIZE) == false) {
      ei_printf("ERR: Could not allocate audio buffer (size %d), this could be due to the window length of your model\r\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT);
      return;
  }



}

/**
 * @brief      Arduino main function. Runs the inferencing loop.
 */
void loop()
{
    bool m = microphone_inference_record();
    if (!m) {
        ei_printf("ERR: Failed to record audio...\n");
        return;
    }

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_SLICE_SIZE;
    signal.get_data = &microphone_audio_signal_get_data;
    ei_impulse_result_t result = {0};

    //EI_IMPULSE_ERROR r = run_classifier_continuous(&signal, &result, debug_nn);

    //Serial.println(ei_default_impulse.impulse->dsp_blocks[0].n_output_features);
    
    if (!outputMatrix.buffer) {
        ei_printf("allocation of output matrix failed\n");
    }
    run_mfcc_maaajaaa(&signal, &outputMatrix, debug_nn);

    /*if (r != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", r);
        return;
    }*/

    double rMax = -100.0;
    int rMaxIndex = -1;
    double gMax = -100.0;
    int gMaxIndex = -1;
    double bMax = -100.0;
    int bMaxIndex = -1;

    //relevant buffer area where the mfcc output is stored
    int relevantBuferCols = mfe_buffer_size.cols;

    int firstThird, secondThird;
    firstThird =  3;
    secondThird = 5;

    
    //print graph, can't be viewed in arduino viewer, but putty or cutecom do support the clear screen command
    //stackoverflow.com/a/15559322
    if(printGraph  && nonPrintCycles >= printEvery){
      Serial.write(27); //ESC
      Serial.print("[2J"); //clear screen
      Serial.write(27); //ESC
      Serial.print("[H"); //cursor to home
    }
    if(!printGraph){
      Serial.print("output: ");
    }
    for(int i = 0; i < relevantBuferCols; i++){
      //find maxima of the thrids of the spectrum
      if(i<firstThird){
        if(outputMatrix.buffer[i] > rMax){
          rMax = outputMatrix.buffer[i];
          rMaxIndex = i;
        }
      }else if(i<secondThird){
        if(outputMatrix.buffer[i] > gMax){
          gMax = outputMatrix.buffer[i];
          gMaxIndex = i;
        }
      }else{
        if(outputMatrix.buffer[i] > bMax){
          bMax = outputMatrix.buffer[i];
          bMaxIndex = i;
        }
      }

      if(!printGraph && i<20){
        Serial.print(outputMatrix.buffer[i]);
        Serial.print(" ");
      } 

      //print graph bar
      if(printGraph && nonPrintCycles >= printEvery){
        for(int j = 0; j<round(graphMaxLength * outputMatrix.buffer[i]); j++){
          Serial.print("▮");
        }
        Serial.println();
      }
      if(i==ceptrumToShow && outputMode == SINGLE_CEPTRUM ){
        for(int j = 0; j<NUMPIXELS; j++){  
          if(j<= round(NUMPIXELS * outputMatrix.buffer[i])){
            pixels.setPixelColor(j,255, 0, 0);
          }else{
            pixels.setPixelColor(j,0, 0, 0);
          }
        }
      }

    }
    if(!printGraph)
      Serial.print("\n");

    if(printGraph && nonPrintCycles >= printEvery){
        nonPrintCycles = 0;
    }else{
      nonPrintCycles++;
    }

    //calculate new 8-bit rbg values, assuming mfcc output is normed to 0..1
    int rNew = pow(rMax,2)*0.6;
    int gNew = pow(gMax,2)*0.6;
    int bNew = pow(bMax,2)*0.6;

    if(rMax < 0.4){
      rNew = 0;
    }
    if(gMax < 0.5){
      gNew = 0;
    }
    if(bMax < 0.3){
      bNew = 0;
    }

    if(!printGraph){

      Serial.print("new rgb: ");
      Serial.print(rNew);
      Serial.print(" ");
      Serial.print(gNew);
      Serial.print(" ");
      Serial.println(bNew);

      Serial.print("max rgb: ");
      Serial.print(rMax);
      Serial.print(" ");
      Serial.print(gMax);
      Serial.print(" ");
      Serial.println(bMax);

      Serial.print("max rgb index: ");
      Serial.print(rMaxIndex);
      Serial.print(" ");
      Serial.print(gMaxIndex);
      Serial.print(" ");
      Serial.println(bMaxIndex);
      
      
    }
    //make copy of pixel array (needed for filtering only)
    std::copy(pixelArray, pixelArray+NUMPIXELS, pixelArrayOld);
    switch(outputMode){

      case LINE_CASCADING:
        //cascade
        for(int i=NUMPIXELS; i>0; i--){
          pixelArray[i] = pixelArray[i-1];
        }
        //set 0th pixel
        pixelArray[0] = {rNew, gNew, bNew};
      break;

      case SYMMETRIC_CASCADING:

        int centerPixel = NUMPIXELS/2+1;
        //cneter to left cascading
        for(int i=NUMPIXELS; i>centerPixel; i--){
          pixelArray[i] = pixelArray[i-1];
          //pixels.setPixelColor(i,pixels.getPixelColor(i-1));
        }
        //center to right cascading
        for(int i=0; i<centerPixel; i++){
          pixelArray[i] = pixelArray[i+1];
          //pixels.setPixelColor(i,pixels.getPixelColor(i+1));
        }
        //set center pixel
        pixelArray[centerPixel] = {rNew, gNew, bNew};
      break;

    }
    //apply filter and apply array to pixels
    //caching arrays to use 3 separate 1d gaussian blur fliters
    float reds[NUMPIXELS];
    float greens[NUMPIXELS];
    float blues[NUMPIXELS];
    for(int i = 0; i<NUMPIXELS; i++){
      //apply low pass filter
      RGBColour filteredCol = filterRGBColour(pixelArray[i], pixelArrayOld[i]);
      reds[i] = filteredCol.r;
      greens[i] = filteredCol.g;
      blues[i] = filteredCol.b;

      //pixels.setPixelColor(i,pixelArray[i].r, pixelArray[i].g, pixelArray[i].b);
      //pixels.setPixelColor(i,filteredCol.r, filteredCol.g, filteredCol.b);
    }

    for(int i = 0; i<NUMPIXELS; i++){
      int r = round(makeAndApplyKernelFromKernelCache(kernelCache,NUMPIXELS, i, reds));
      int g = round(makeAndApplyKernelFromKernelCache(kernelCache,NUMPIXELS, i, greens));
      int b = round(makeAndApplyKernelFromKernelCache(kernelCache,NUMPIXELS, i, blues));
      pixels.setPixelColor(i,r,g,b);
    }
    pixels.show();   // Send the updated pixel colors to the hardware.
}

/**
 * @brief      PDM buffer full callback
 *             Get data and call audio thread callback
 */
static void pdm_data_ready_inference_callback(void)
{
    int bytesAvailable = PDM.available();

    // read into the sample buffer
    int bytesRead = PDM.read((char *)&sampleBuffer[0], bytesAvailable);

    if (record_ready == true) {
        for (int i = 0; i<bytesRead>> 1; i++) {
            inference.buffers[inference.buf_select][inference.buf_count++] = sampleBuffer[i];

            if (inference.buf_count >= inference.n_samples) {
                inference.buf_select ^= 1;
                inference.buf_count = 0;
                inference.buf_ready = 1;
            }
        }
    }
}

/**
 * @brief      Init inferencing struct and setup/start PDM
 *
 * @param[in]  n_samples  The n samples
 *
 * @return     { description_of_the_return_value }
 */
static bool microphone_inference_start(uint32_t n_samples)
{
    inference.buffers[0] = (signed short *)malloc(n_samples * sizeof(signed short));

    if (inference.buffers[0] == NULL) {
        return false;
    }

    inference.buffers[1] = (signed short *)malloc(n_samples * sizeof(signed short));

    if (inference.buffers[1] == NULL) {
        free(inference.buffers[0]);
        return false;
    }

    sampleBuffer = (signed short *)malloc((n_samples >> 1) * sizeof(signed short));

    if (sampleBuffer == NULL) {
        free(inference.buffers[0]);
        free(inference.buffers[1]);
        return false;
    }

    inference.buf_select = 0;
    inference.buf_count = 0;
    inference.n_samples = n_samples;
    inference.buf_ready = 0;

    // configure the data receive callback
    PDM.onReceive(&pdm_data_ready_inference_callback);

    PDM.setBufferSize((n_samples >> 1) * sizeof(int16_t));

    // initialize PDM with:
    // - one channel (mono mode)
    // - a 16 kHz sample rate
    if (!PDM.begin(1, EI_CLASSIFIER_FREQUENCY)) {
        ei_printf("Failed to start PDM!");
    }

    // set the gain, defaults to 20
    PDM.setGain(127);

    record_ready = true;

    return true;
}

/**
 * @brief      Wait on new data
 *
 * @return     True when finished
 */
static bool microphone_inference_record(void)
{
    bool ret = true;

    if (inference.buf_ready == 1) {
        ei_printf(
            "Error sample buffer overrun. Decrease the number of slices per model window "
            "(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW)\n");
        ret = false;
    }

    while (inference.buf_ready == 0) {
        delay(1);
    }

    inference.buf_ready = 0;

    return ret;
}

/**
 * Get raw audio signal data
 */
static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr)
{
    numpy::int16_to_float(&inference.buffers[inference.buf_select ^ 1][offset], out_ptr, length);

    return 0;
}

/**
 * @brief      Stop PDM and release buffers
 */
static void microphone_inference_end(void)
{
    PDM.end();
    free(inference.buffers[0]);
    free(inference.buffers[1]);
    free(sampleBuffer);
}

//Tf is the filter time constant
//Ts ist the sampling time
float lowPassFilter(float alpha, float y, float y_prev){
  //based on simple FOC equation https://docs.simplefoc.com/low_pass_filter
  // calculate the filtering 
  //float alpha = Tf/(Tf + Ts);
  return alpha*y_prev + (1.0f - alpha) * y;
}



RGBColour filterRGBColour(RGBColour rgbColourCurrent, RGBColour rgbColourLast){
  RGBColour rgbColour;
  rgbColour.r = lowPassFilter(alphaLowPass, rgbColourCurrent.r, rgbColourLast.r);
  rgbColour.g = lowPassFilter(alphaLowPass, rgbColourCurrent.g, rgbColourLast.g);
  rgbColour.b = lowPassFilter(alphaLowPass, rgbColourCurrent.b, rgbColourLast.b);
  return rgbColour;
}

//Gaussian Blut 1D, based on https://github.com/Maaajaaa/Gaussian_filter_1D/ which is forked off of https://github.com/lchop/Gaussian_filter_1D_cpp

void computeKernelCache(float *kernelCache, int n_points, float sigma)
{
    //Compute the kernel for the given x point
    //calculate sigma² once to speed up calculation
    float twoSigmaSquared = (2*pow(sigma,2));
    for (int i =0; i<n_points*2;i++)
    {
        //Compute gaussian kernel
        //kernel cache at 0 is -1*n_points
        kernelCache[i] = exp(-(pow(-1*n_points + i,2) / twoSigmaSquared));
    }
    return;
}

float makeAndApplyKernelFromKernelCache(float kernelCache[], int n_points, int x_position, float y_values[])
{
    //make array for the actual kernel for the given x point
    float kernel[n_points] = {};
    float sum_kernel = 0;
    for (int i =0; i<n_points;i++)
    {
        //fetch kernel vale from kernel cache
        //+ n_points as -npoints is at 0
        kernel[i] = kernelCache[i - x_position + n_points];
        //compute a weight for each kernel position
        sum_kernel += kernel[i];
    }
    //apply weight to each kernel position to give more important value to the x that are around ower x
    for(int i = 0;i<n_points;i++)
        kernel[i] = kernel[i] / sum_kernel;
    return applyKernel(n_points, x_position, kernel, y_values);
}

float applyKernel(int n_points, int x_position, float kernel[], float y_values[])
{
    float y_filtered = 0;
    //apply filter to all the y values with the weighted kernel
    for(int i = 0;i<n_points;i++) 
        y_filtered += kernel[i] * y_values[i];

    return y_filtered;
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Invalid model for current sensor."
#endif

