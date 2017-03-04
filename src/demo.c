#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include <sys/time.h>

#define FRAMES 3

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
image get_image_from_stream(CvCapture *cap);
char get_image_from_stream2(CvCapture *cap, IplImage **ipl, image *img);
char get_image_from_stream3(CvCapture *cap, image *rawImg, image *resizedImg, int w, int h);

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static float **probs;
static box *boxes;
static network net;
static IplImage *ipl;
static image in   ;
static image in_s ;
static image det  ;
static image det_s;
static image disp = {0};
static CvCapture * cap;
static CvVideoWriter *writer;
static float frameTime;
static float fps = 0;
static float demo_thresh = 0;
static float demo_hier_thresh = .5;


static float *predictions[FRAMES];
static int demo_index = 0;
static image images[FRAMES];
static float *avg; 


void *fetch_in_thread(void *ptr)
{
    if (!get_image_from_stream2(cap, &ipl, &in_s))
    {
        error("Stream closed.");
    }    
    in = in_s;
    // if (!get_image_from_stream3(cap, &in, &in_s, net.w, net.h))
    // {
    //     error("Stream closed.");
    // }
    return 0;
}

void *detect_in_thread(void *ptr)
{
    // printf("\033[2J");
    // printf("\033[1;1H");
    static double detectThreadTimings;
    static int count;
    count++;
    // int idx = 0;
    double t1, t2;
    t1 = get_wall_time_us();
    float nms = .4;

    layer l = net.layers[net.n-1];
    float *X = det_s.data;
    float *prediction = network_predict(net, X);

    t2 = get_wall_time_us();
    detectThreadTimings += t2-t1;
    t1 = t2;

    memcpy(predictions[demo_index], prediction, l.outputs*sizeof(float));
    mean_arrays(predictions, FRAMES, l.outputs, avg);
    l.output = avg;

    // t2 = get_wall_time_us();
    // detectThreadTimings[idx++] += t2-t1;
    // t1 = t2;

    // free_image(det_s);
    if(l.type == DETECTION){
        get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
    } else if (l.type == REGION){
        get_region_boxes(l, 1, 1, demo_thresh, probs, boxes, 1, 0, demo_hier_thresh);
    } else {
        error("Last layer must produce detections\n");
    }

    // t2 = get_wall_time_us();
    // detectThreadTimings[idx++] += t2-t1;
    // t1 = t2;

    if (nms > 0) do_nms(boxes, probs, l.w*l.h*l.n, l.classes, nms);

    // t2 = get_wall_time_us();
    // detectThreadTimings[idx++] += t2-t1;
    // t1 = t2;

    printf("FPS:%.1f %05d(%03d)ms\n\n", fps, (int)detectThreadTimings/1000, (int)detectThreadTimings/count/1000);
    // printf("\tnetwork_predict %05d(%03d)\n", (int)detectThreadTimings[0]/1000, 
    //     (int)detectThreadTimings[0]/count/1000);
    // printf("\tmean_arrays %05d(%03d)\n", (int)detectThreadTimings[1]/1000, 
    //     (int)detectThreadTimings[1]/count/1000);
    // printf("\tget_region_boxes %05d(%03d)\n", (int)detectThreadTimings[2]/1000, 
    //     (int)detectThreadTimings[2]/count/1000);
    // printf("\tdo_nms %05d(%03d)\n", (int)detectThreadTimings[3]/1000, 
    //     (int)detectThreadTimings[3]/count/1000);
    // printf("Objects:\n\n");

    images[demo_index] = det;
    det = images[(demo_index + FRAMES/2 + 1)%FRAMES];
    demo_index = (demo_index + 1)%FRAMES;

    // draw_detections(det, l.w*l.h*l.n, demo_thresh, boxes, probs, demo_names, demo_alphabet, demo_classes);

    int i = 0;
    while (i < l.w*l.h*l.n) {
        // int row = i / l.w;
        // int col = i % l.w;
        float prob = probs[i][0];
        box b = boxes[i];
        i++;
        if (prob < demo_thresh) {
            continue;
        }

        int x1  = (b.x-b.w/2.)*320;
        int y1  = (b.y-b.h/2.)*240;
        int x2  = (b.x+b.w/2.)*320;
        int y2  = (b.y+b.h/2.)*240;

        // printf("%d, %d, %f, (%.0f,%.0f) (%.0f,%.0f)\n", row, col, prob, b.x*320, b.y*240, b.w*320, b.h*240);

        int prob255 = 255*prob;
        int bValue = 255 - prob255;
        int gValue = prob255 < 128 ? prob255 * 2 : (unsigned char)((255 - prob255) * 2);
        int rValue = prob255;
        cvRectangle(ipl, cvPoint(x1, y1), cvPoint(x2, y2), CV_RGB(rValue, gValue, bValue), 1, 8, 0);

        // char str[10];
        // sprintf(str, "%.1f%%", probs[i][0]*100);
        // CvFont font;    
        // cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.2, 0.2, 1, 1, 8);  
        // cvPutText(ipl, str, cvPoint(x1, y2), &font, CV_RGB(255, 0, 0));
    }

    return 0;
}

double get_wall_time()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix, float hier_thresh)
{
    //skip = frame_skip;
    image **alphabet = load_alphabet();
    int delay = frame_skip;
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_hier_thresh = hier_thresh;
    printf("Demo\n");
    net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);

    srand(2222222);

    if(filename){
        printf("video file: %s\n", filename);
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);
    }
    if(!cap) error("Couldn't connect to webcam.\n");

    writer = cvCreateVideoWriter("out.avi", CV_FOURCC('M','J','P','G'), 20, cvSize(320, 240), 1);
    if (!writer) error("Couldn't create video writer\n");

    layer l = net.layers[net.n-1];
    int j;

    avg = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < FRAMES; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < FRAMES; ++j) images[j] = make_image(1,1,3);

    boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
    probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float));

    pthread_t fetch_thread;
    pthread_t detect_thread;

    // fetch_in_thread(0);
    // det = in;
    // det_s = in_s;

    // fetch_in_thread(0);
    // detect_in_thread(0);
    // disp = det;
    // det = in;
    // det_s = in_s;

    // for(j = 0; j < FRAMES/2; ++j){
    //     fetch_in_thread(0);
    //     detect_in_thread(0);
    //     disp = det;
    //     det = in;
    //     det_s = in_s;
    // }
    // in = make_image(320, 240, 3);
    in_s = make_zero_copy_image(net.w, net.h, 3);

    int count = 0;
    // if(!prefix){
    //     cvNamedWindow("Demo", CV_WINDOW_AUTOSIZE); 
    //     cvMoveWindow("Demo", 0, 0);
    //     // cvResizeWindow("Demo", 1352, 1013);
    // }

    double before = get_wall_time();

    while(1){
        ++count;
        if(0){
            if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
            if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");

            if(!prefix){
                show_image(disp, "Demo");
                int c = cvWaitKey(1);
                if (c == 10) {
                    cvWaitKey(-1);
                    // if(frame_skip == 0) frame_skip = 60;
                    // else if(frame_skip == 4) frame_skip = 0;
                    // else if(frame_skip == 60) frame_skip = 4;   
                    // else frame_skip = 0;
                }
            }else{
                char buff[256];
                sprintf(buff, "%s_%08d", prefix, count);
                save_image(disp, buff);
            }

            pthread_join(fetch_thread, 0);
            pthread_join(detect_thread, 0);

            if(delay == 0){
                free_image(disp);
                disp  = det;
            }
            det   = in;
            det_s = in_s;
        }else {
            double t1, t2;
            t1 = get_wall_time_ms();
            fetch_in_thread(0);
            t2 = get_wall_time_ms();
            printf("\t\tfetch_in_thread %d\n", (int)(t2-t1));
            t1 = t2;
            det   = in;
            det_s = in_s;
            t2 = get_wall_time_ms();
            printf("\t\t set image %d\n", (int)(t2-t1));
            t1 = t2;
            detect_in_thread(0);
            t2 = get_wall_time_ms();
            printf("\t\t detect_in_thread %d\n", (int)(t2-t1));
            t1 = t2;
            // disp = det;
            cvWriteFrame(writer, ipl);
            // show_image(det, "Demo");
            // cvWaitKey(1);
            t2 = get_wall_time_ms();
            printf("\t\t cvWriteFrame %d\n", (int)(t2-t1));
            t1 = t2;
        }
        // --delay;
#define FPS_AVERAGE_FRAME_NUM 10
        if(count % FPS_AVERAGE_FRAME_NUM == 0){
            // delay = frame_skip;

            double after = get_wall_time();
            frameTime = after - before;
            float curr = FPS_AVERAGE_FRAME_NUM/frameTime;
            fps = curr;
            before = after;
        }
        if (count >= 1000)
            error("enough\n");
    }
}
#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix, float hier_thresh)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif

