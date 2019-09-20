/**
 * @file Tracking.cc
 * @author guoqing (1337841346@qq.com)
 * @brief 追踪线程
 * @version 0.1
 * @date 2019-02-21
*/


#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"

#include"Optimizer.h"
#include"PnPsolver.h"

#include<iostream>
#include<cmath>
#include<mutex>


using namespace std;

// 程序中变量名的第一个字母如果为"m"则表示为类中的成员变量，member
// 第一个、第二个字母:
// "p"表示指针数据类型
// "n"表示int类型
// "b"表示bool类型
// "s"表示set类型
// "v"表示vector数据类型
// 'l'表示list数据类型
// "KF"表示KeyPoint数据类型   NOTICE

namespace ORB_SLAM2
{

// **************************************************************************************************************************************************
// ********************************************************* 构造函数 ********************************************************************************
// **************************************************************************************************************************************************
Tracking::Tracking(
    System *pSys,                       //系统实例
    ORBVocabulary* pVoc,                //BOW字典
    FrameDrawer *pFrameDrawer,          //帧绘制器
    MapDrawer *pMapDrawer,              //地图点绘制器
    Map *pMap,                          //地图句柄
    KeyFrameDatabase* pKFDB,            //关键帧产生的词袋数据库
    const string &strSettingPath,       //配置文件路径
    const int sensor):                  //传感器类型
        mState(NO_IMAGES_YET),                              //当前系统还没有准备好
        mSensor(sensor),                                
        mbOnlyTracking(false),                              //处于SLAM模式
        mbVO(false),                                        //当处于纯跟踪模式的时候，这个变量表示了当前跟踪状态的好坏
        mpORBVocabulary(pVoc),          
        mpKeyFrameDB(pKFDB), 
        mpInitializer(static_cast<Initializer*>(NULL)),     //暂时给地图初始化器设置为空指针
        mpSystem(pSys), 
        mpViewer(NULL),                                     //注意可视化的查看器是可选的，因为ORB-SLAM2最后是被编译成为一个库，
                                    // 所以对方人拿过来用的时候也应该有权力说我不要可视化界面（何况可视化界面也要占用不少的CPU资源）
        mpFrameDrawer(pFrameDrawer),
        mpMapDrawer(pMapDrawer), 
        mpMap(pMap), 
        mnLastRelocFrameId(0)                               //恢复为0,没有进行这个过程的时候的默认值
{
    // step1************************************************从配置文件加载相机参数**********************************************

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    //     |fx  0   cx|
    // K = |0   fy  cy|
    //     |0   0   1 |
    //构造相机内参矩阵K
    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    // 图像矫正系数，畸变矩阵
    // [k1 k2 p1 p2 k3]
    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    //有些相机的畸变系数中会没有k3项
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    // 双目摄像头baseline * fx 50
    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // 插入关键帧和检查重定位的最多/最少帧
    mMinFrames = 0;
    mMaxFrames = fps;

    //输出
    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;

    // 1:RGB 0:BGR
    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

// step2******************************************从配置文件加载ORB字典参数*******************************************************

    // 每一帧提取的特征点数 1000
    int nFeatures = fSettings["ORBextractor.nFeatures"];
    // 图像建立金字塔时的变化尺度 1.2
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    // 尺度金字塔的层数 8
    int nLevels = fSettings["ORBextractor.nLevels"];
    // 提取fast特征点的默认阈值 20
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    // 如果默认阈值提取不出足够fast特征点，则使用最小阈值 8
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    // tracking过程都会用到mpORBextractorLeft作为特征点提取器
    mpORBextractorLeft = new ORBextractor(
        nFeatures,      //参数的含义还是看上面的注释吧
        fScaleFactor,
        nLevels,
        fIniThFAST,
        fMinThFAST);

    // 如果是双目，tracking过程中还会用用到mpORBextractorRight作为右目特征点提取器
    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    // 在单目初始化的时候，会用mpIniORBextractor来作为特征点提取器
    if(sensor==System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        // 判断一个3D点远/近的阈值 mbf * 35 / fx
        //ThDepth其实就是表示基线长度的多少倍
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if(sensor==System::RGBD)
    {
        // 深度相机disparity转化为depth时的因子
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

}
// **************************************************************************************************************************************************
// ********************************************************* 修改标志的函数 ***************************************************************************
// **************************************************************************************************************************************************
//设置局部建图器
void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

//设置回环检测器
void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

//设置可视化查看器
void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}

// **************************************************************************************************************************************************
// ********************************************************* 处理单目数据的函数 ************************************************************************
// **************************************************************************************************************************************************
// 输入左右目图像，可以为RGB、BGR、RGBA、GRAY
// 1、将图像转为mImGray和imGrayRight并初始化mCurrentFrame
// 2、进行tracking过程
// 输出世界坐标系到该帧相机坐标系的变换矩阵
cv::Mat Tracking::GrabImageStereo(
    const cv::Mat &imRectLeft,      //左侧图像
    const cv::Mat &imRectRight,     //右侧图像
    const double &timestamp)        //时间戳
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    // step 1 ：将RGB或RGBA图像转为灰度图像
    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    //NOTE 这里考虑得十分周全,甚至连四通道的图像都考虑到了
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }

    // step 2 ：构造Frame
    mCurrentFrame = Frame(
        mImGray,                //左目图像
        imGrayRight,            //右目图像
        timestamp,              //时间戳
        mpORBextractorLeft,     //左目特征提取器
        mpORBextractorRight,    //右目特征提取器
        mpORBVocabulary,        //字典
        mK,                     //内参矩阵
        mDistCoef,              //去畸变参数
        mbf,                    //基线长度
        mThDepth);              //远点,近点的区分阈值

    // step 3 ：跟踪
    Track();

    //返回位姿
    return mCurrentFrame.mTcw.clone();
}

// **************************************************************************************************************************************************
// ********************************************************* 处理RGB-D数据的函数 **********************************************************************
// **************************************************************************************************************************************************
// 输入左目RGB或RGBA图像和深度图
// 1、将图像转为mImGray和imDepth并初始化mCurrentFrame
// 2、进行tracking过程
// 输出世界坐标系到该帧相机坐标系的变换矩阵
cv::Mat Tracking::GrabImageRGBD(
    const cv::Mat &imRGB,           //彩色图像
    const cv::Mat &imD,             //深度图像
    const double &timestamp)        //时间戳
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    // step 1：将RGB或RGBA图像转为灰度图像
    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    // step 2 ：将深度相机的disparity转为Depth , 也就是转换成为真正尺度下的深度
    //这里的判断条件感觉有些尴尬
    //前者和后者满足一个就可以了
    //满足前者意味着,mDepthMapFactor 相对1来讲要足够大
    //满足后者意味着,如果深度图像不是浮点型? 才会执行
    //意思就是说,如果读取到的深度图像是浮点型,就不执行这个尺度的变换操作了呗?
    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(  //将图像转换成为另外一种数据类型,具有可选的数据大小缩放系数
            imDepth,            //输出图像
            CV_32F,             //输出图像的数据类型
            mDepthMapFactor);   //缩放系数

    // 步骤3：构造Frame
    mCurrentFrame = Frame(
        mImGray,                //灰度图像
        imDepth,                //深度图像
        timestamp,              //时间戳
        mpORBextractorLeft,     //ORB特征提取器
        mpORBVocabulary,        //词典
        mK,                     //相机内参矩阵
        mDistCoef,              //相机的去畸变参数
        mbf,                    //相机基线*相机焦距
        mThDepth);              //内外点区分深度阈值

    // 步骤4：跟踪
    Track();

    //返回当前帧的位姿
    return mCurrentFrame.mTcw.clone();
}

// **************************************************************************************************************************************************
// ********************************************************* 处理双目数据的函数 ************************************************************************
// **************************************************************************************************************************************************
// 输入左目RGB或RGBA图像
// 1、将图像转为mImGray并初始化mCurrentFrame
// 2、进行tracking过程
// 输出世界坐标系到该帧相机坐标系的变换矩阵
cv::Mat Tracking::GrabImageMonocular(
    const cv::Mat &im,          //单目图像
    const double &timestamp)    //时间戳
{

    mImGray = im;

    // step 1 ：将RGB或RGBA图像转为灰度图像
    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    // step 2 ：构造Frame
    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)// 没有成功初始化的前一个状态就是NO_IMAGES_YET
        mCurrentFrame = Frame(
            mImGray,
            timestamp,
            mpIniORBextractor,      //LNOTICE 这个使用的是初始化的时候的ORB特征点提取器
            mpORBVocabulary,
            mK,
            mDistCoef,
            mbf,
            mThDepth);
    else
        mCurrentFrame = Frame(
            mImGray,
            timestamp,
            mpORBextractorLeft,     //NOTICE 当程序正常运行的时候使用的是正常的ORB特征点提取器
            mpORBVocabulary,
            mK,
            mDistCoef,
            mbf,
            mThDepth);

    // step 3 ：跟踪
    Track();
    //返回当前帧的位姿
    return mCurrentFrame.mTcw.clone();
}

// **************************************************************************************************************************************************
// ********************************************************* Track函数：执行VO和局部地图更新，或者仅定位 *************************************************
// **************************************************************************************************************************************************
// NOTICE 敲黑板: Track包含两部分：估计运动、跟踪局部地图
void Tracking::Track()
{
    // mState为tracking的状态标记
    // SYSTME_NOT_READY, NO_IMAGE_YET, NOT_INITIALIZED, OK, LOST
    // 如果图像复位过、或者第一次运行，则为NO_IMAGE_YET状态
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState;// mLastProcessedState 存储了Tracking最新的状态，用于FrameDrawer中的绘制

    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);//上锁，其实也就只有在这里才上了一次锁

    // step1：如果没有初始化，现在就来初始化
    if(mState==NOT_INITIALIZED)
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD)
            StereoInitialization();//双目初始化(RGBD相机也当做为双目相机来看待)
        else
            MonocularInitialization();//单目初始化

        mpFrameDrawer->Update(this);//这一帧处理完成了,更新帧绘制器中存储的最近的一份状态

        if(mState!=OK)//这个状态量在上面的初始化函数中被更新了，所以检查一下
            return;
    }
    // step2：已完成初始化，就执行跟踪
    else
    {
        bool bOK;// bOK为临时变量，用于表示每个函数是否执行成功

        // 如果lost，就使用运动模型或重定位来初始化相机位姿

        if(!mbOnlyTracking)//VO模式
        {
            if(mState==OK)// 正常初始化成功
            {
                // 检查并更新上一帧被替换的MapPoints
                // 更新Fuse函数和SearchAndFuse函数替换的MapPoints
                //由于追踪线程需要使用上一帧的信息,而局部建图线程则可能会对原有的地图点进行替换.在这里进行检查
                CheckReplacedInLastFrame();

                if(mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2)// 匀速运动模型是空的或刚完成重定位
                {
                    // 将上一帧的位姿作为当前帧的初始位姿，优化重投影误差来定位
                    bOK = TrackReferenceKeyFrame();
                }
                else//根据匀速运动模型设定当前帧的初始位姿
                {
                    bOK = TrackWithMotionModel();
                    if(!bOK)//如果失败了，还是用上面的方法来跟踪
                        bOK = TrackReferenceKeyFrame();
                }
            }
            else//如果没有正常初始化，就重定位
            {
                bOK = Relocalization();
            }
        }
        else//仅定位模式
        {
            // tracking跟丢了, 那么就只能进行重定位了
            if(mState==LOST)
            {
                bOK = Relocalization();
            }
            else// 如果没有跟丢
            {
                // mbVO是mbOnlyTracking为true时才有的一个变量
                if(!mbVO)//mbVO为false表示此帧匹配了很多的MapPoints，跟踪很正常
                {
                    if(!mVelocity.empty())
                    {
                        bOK = TrackWithMotionModel();//匀速运动模型可用，采用匀速运动模型跟踪
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();//优化重投影误差来跟踪
                    }
                }
                else// mbVO为true表明此帧匹配了很少的MapPoints，少于10个，要跪的节奏
                {
                    // 我们计算了两个相机姿势，一个来自运动模型，另一个来自重新定位。（而跟踪正常的时候，首选运动模型，其次优化重投影误差）
                    // 如果重新定位成功，我们选择该解，否则我们保留VO解

                    bool bOKMM = false;//MM=Motion Model,通过运动模型进行跟踪的结果
                    bool bOKReloc = false;//通过重定位方法来跟踪的结果
                    vector<MapPoint*> vpMPsMM;//运动模型中构造的地图点
                    vector<bool> vbOutMM;//在追踪运动模型后发现的外点
                    cv::Mat TcwMM;//运动模型得到的位姿

                    if(!mVelocity.empty())//当运动模型非空的时候,根据运动模型计算位姿
                    {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }

                    bOKReloc = Relocalization();//使用重定位的方法来得到当前帧的位姿

                    // 重定位没有成功，而运动模型跟踪成功
                    if(bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        //如果重定位本身要跪了
                        if(mbVO)//匹配点很少（其实不用再判断了，必然事件）
                        {
                            // 这段代码是不是有点多余？应该放到 TrackLocalMap 函数中统一做
                            for(int i =0; i<mCurrentFrame.N; i++)// 更新当前帧的MapPoints被观测程度
                            {
                                //如果这个特征点形成了地图点,并且也不是外点的时候
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    //增加被观测的次数
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if(bOKReloc)
                    {
                        mbVO = false;// 只要重定位成功整个跟踪过程正常进行（定位与跟踪，更相信重定位）
                    }
                    bOK = bOKReloc || bOKMM;//有一个成功我们就认为执行成功了
                }
            }
        }

        // 将最新的关键帧作为reference frame
        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // 上面的操作：运动模型跟踪、重定位、优化重投影，都是为了得到一个位姿的初始估计
        // 下面根据这个初始估计的位姿来进行更精确的定位

        //跟踪模式
        if(!mbOnlyTracking)
        {
            if(bOK)//如果估计初始位姿成功，就进行更的精确定位
                bOK = TrackLocalMap();
        }
        else
        {
            // 估计初始位姿失败，但是重定位成功，也继续更精确的定位
            if(bOK && !mbVO)
                bOK = TrackLocalMap();
        }

        if(bOK)//根据上面的操作来判断是否追踪成功
            mState = OK;
        else
            mState=LOST;

        mpFrameDrawer->Update(this);// Update drawer 中的帧副本的信息

        // 清除跟踪时添加的点、判断是否添加关键帧、并执行添加
        if(bOK)
        {
            // 更新匀速运动模型
            if(!mLastFrame.mTcw.empty())
            {
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                //? 这个是转换成为了相机相对世界坐标系的旋转?
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                mVelocity = mCurrentFrame.mTcw*LastTwc; // 其实就是 Tcl
            }
            else
                //否则生成空矩阵
                mVelocity = cv::Mat();

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            //清除UpdateLastFrame中为当前帧临时添加的MapPoints
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    // 清除UpdateLastFrame函数中为了跟踪增加的MapPoints
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            //清除临时的MapPoints，这些MapPoints在 TrackWithMotionModel 的 UpdateLastFrame 函数里生成（仅双目和rgbd）
            // 前面只是在当前帧中将这些MapPoints剔除，这里从MapPoints数据库中删除
            // 这里生成的仅仅是为了提高双目或rgbd摄像头的帧间跟踪效果，用完以后就扔了，没有添加到地图中
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;// 这里不仅仅是清除mlpTemporalPoints，通过delete pMP还删除了指针指向的MapPoint
            }
            mlpTemporalPoints.clear();//不能够直接执行这个是因为其中存储的都是指针,之前的操作都是为了避免内存泄露

            // 检查是否要插入新关键帧，NOTICE 在插入关键帧时生成地图点
            if(NeedNewKeyFrame())
                CreateNewKeyFrame();

            //删除那些在BA中检测为outlier的3D map点
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                //这里第一个条件还要执行判断是因为, 前面的操作中可能删除了其中的地图点
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        // 跟踪失败，而且关键帧很少，重新Reset
        if(mState==LOST)
        {
            if(mpMap->KeyFramesInMap()<=5)
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
        }

        //确保已经设置了参考关键帧
        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);// 保存上一帧的数据,当前帧变上一帧
    }
    //跟踪过程完成

    // 记录位姿信息及各种状态
    if(!mCurrentFrame.mTcw.empty())
    {

        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();// 当前帧位姿
        //保存各种状态
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else
    {
        // 如果跟踪失败，则相对位姿使用上一次值
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }

}// Tracking 

// **************************************************************************************************************************************************
// ******************************************************************* 双目和rgbd的地图初始化 **********************************************************
// **************************************************************************************************************************************************
/*
 * @brief 双目和rgbd的地图初始化
 *
 * 由于具有深度信息，直接生成MapPoints
 */
void Tracking::StereoInitialization()
{
    //整个函数只有在当前帧的特征点超过500的时候才会进行
    if(mCurrentFrame.N>500)
    {
        // step 1：设定初始位姿
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

        // step 2：将当前帧构造为初始关键帧
        // KeyFrame包含Frame、地图3D点、以及BoW
        // KeyFrame里的mpMap都指向Tracking里的mpMap
        // KeyFrame里的mpMap都指向Tracking里的mpKeyFrameDB
        // 提问: 为什么要指向Tracking中的相应的变量呢? -- 因为Tracking是主线程，是它创建和加载的这些模块
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

        // step 3：在地图中添加该初始关键帧
        // KeyFrame中包含了地图、反过来地图中也包含了KeyFrame
        mpMap->AddKeyFrame(pKFini);

        // step 4：为每个特征点构造MapPoint
        for(int i=0; i<mCurrentFrame.N;i++)
        {
            //只有具有正深度的点才会被构造地图点
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);// step 4.1：通过反投影得到该特征点的3D坐标
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);// step 4.2：将3D点构造为MapPoint

                // step 4.3：为该MapPoint添加属性：
                // a.观测到该MapPoint的关键帧
                // b.该MapPoint的描述子
                // c.该MapPoint的平均观测方向和深度范围

                pNewMP->AddObservation(pKFini,i);// a.表示该MapPoint可以被哪个KeyFrame的哪个特征点观测到
                pNewMP->ComputeDistinctiveDescriptors();// b.从众多观测到该MapPoint的特征点中挑选区分度最高的描述子
                pNewMP->UpdateNormalAndDepth();// c.更新该MapPoint平均观测方向以及观测距离的范围

                mpMap->AddMapPoint(pNewMP);// step 4.4：在地图中添加该MapPoint
                pKFini->AddMapPoint(pNewMP,i);// step 4.5：表示该KeyFrame的哪个特征点可以观测到哪个3D点

                mCurrentFrame.mvpMapPoints[i]=pNewMP;// step 4.6：将该MapPoint添加到当前帧的mvpMapPoints中，为当前Frame的特征点与MapPoint之间建立索引
            }
        }

        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

        // step 4：在局部地图中添加该初始关键帧
        mpLocalMapper->InsertKeyFrame(pKFini);

        //当前帧变上一帧
        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;


        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpMap->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        // 把当前（最新的）局部MapPoints作为ReferenceMapPoints
        // ReferenceMapPoints是DrawMapPoints函数画图的时候用的
        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
        //? 设置原点?
        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        //追踪成功
        mState=OK;
    }
}

// **************************************************************************************************************************************************
// ******************************************************************* 单目地图地图初始化 **************************************************************
// **************************************************************************************************************************************************
/*
 * @brief 单目的地图初始化
 *
 * 并行地计算基础矩阵和单应性矩阵，选取其中一个模型，恢复出最开始两帧之间的相对姿态以及点云
 * 得到初始两帧的匹配、相对运动、初始MapPoints
 */
void Tracking::MonocularInitialization()
{
    if(!mpInitializer)// 如果单目初始器还没有被创建，则创建单目初始器
    {
        // Set Reference Frame
        if(mCurrentFrame.mvKeys.size()>100)// 单目初始帧的特征点数必须大于100
        {
            mInitialFrame = Frame(mCurrentFrame);// step 1：得到用于初始化的第一帧，初始化需要两帧
            mLastFrame = Frame(mCurrentFrame);// 记录最近的一帧

            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());// mvbPrevMatched最大的情况就是所有特征点都被跟踪上
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            if(mpInitializer)//可删，必然事件
                delete mpInitializer;

            // 由当前帧构造初始器
            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);

            // -1 表示没有任何匹配。这里面存储的是匹配的点的id
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);//将一个区间的元素都赋予value值

            return;
        }
    }
    else //如果单目初始化器已经被创建
    {
        // step 2：如果当前帧特征点数大于100，则得到用于单目初始化的第二帧，计算两帧的匹配点数
        // 匹配成功点数>100时，可以完成初始化
        // NOTICE 因此只有连续两帧的特征点个数都大于100时，才能继续成功执行Initialize()

        if((int)mCurrentFrame.mvKeys.size()<=100)//第一帧的特征点数<100，直接结束了
        {
            // 重新构造初始器
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            return;
        }

        // step 3：在两帧之间找匹配的特征点对
        ORBmatcher matcher(
            0.9,        //最佳的和次佳评分的比值阈值
            true);      //检查特征点的方向

        //针对单目初始化的时候,进行特征点的匹配
        int nmatches = matcher.SearchForInitialization(
            mInitialFrame,mCurrentFrame,//参考帧和当前帧
            mvbPrevMatched,                 //在初始化参考帧中提取得到的特征点
            mvIniMatches,                   //存储mInitialFrame,mCurrentFrame之间匹配的特征点
            100);                    //搜索窗口大小

        // step 4：如果初始化的两帧之间的匹配点太少，重新初始化
        if(nmatches<100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            return;
        }

        cv::Mat Rcw;
        cv::Mat tcw;
        vector<bool> vbTriangulated;

        // step 5：通过H模型或F模型进行单目初始化，得到两帧间相对运动、初始MapPoints
        if(mpInitializer->Initialize(//得到有足够匹配的连续帧后，执行单目初始化（Initializer.cpp），若初始化成功就继续执行本函数
            mCurrentFrame,      //当前帧
            mvIniMatches,       //当前帧和参考帧的特征点的匹配关系
            Rcw, tcw,           //初始化得到的相机的位姿
            mvIniP3D,           //进行三角化得到的空间点集合
            vbTriangulated))    //以及对应于mvIniMatches来讲,其中哪些点被三角化了
        {
            //当初始化成功的时候
            // step 6：删除那些无法进行三角化的匹配点
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            // 将初始化的第一帧作为世界坐标系，因此第一帧变换矩阵为单位矩阵
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            // 由Rcw和tcw构造Tcw,并赋值给mTcw，mTcw为世界坐标系到该帧的变换矩阵
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);

            // step 6：将三角化得到的3D点包装成MapPoints
            // Initialize函数会得到mvIniP3D，
            // mvIniP3D是cv::Point3f类型的一个容器，是个存放3D点的临时变量，
            // CreateInitialMapMonocular将3D点包装成MapPoint类型存入KeyFrame和Map中
            CreateInitialMapMonocular();
        }
    }
}

// **************************************************************************************************************************************************
// ************************************************************* 单目初始化时生成MapPoints ************************************************************
// **************************************************************************************************************************************************
/*
 * @brief CreateInitialMapMonocular
 *
 * 为单目摄像头初始化得到的点生成MapPoints
 */
//使用在单目初始化过程中三角化得到的点,包装成为地图点,并且生成初始地图
void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames 认为单目初始化时的参考帧和当前帧都是关键帧
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB);  // 第一帧
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);  // 第二帧


    pKFini->ComputeBoW();// step 1：将初始关键帧的描述子转为BoW
    pKFcur->ComputeBoW();// step 2：将当前关键帧的描述子转为BoW

    // step 3：将关键帧插入到地图
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // step 4：将3D点包装成MapPoints，并与关键帧关联
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        cv::Mat worldPos(mvIniP3D[i]);//空间点的世界坐标

        // step 4.1：用3D点构造MapPoint
        MapPoint* pMP = new MapPoint(
            worldPos,
            pKFcur,     // 初始帧是空的，第2帧才用来存放地图点
            mpMap);

        // step 4.2：为该MapPoint添加属性：
        // a.观测到该MapPoint的关键帧
        // b.该MapPoint的描述子
        // c.该MapPoint的平均观测方向和深度范围

        // step 4.3：表示该KeyFrame的哪个特征点可以观测到哪个3D点
        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        // a.表示该MapPoint可以被哪个KeyFrame的哪个特征点观测到
        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        // b.从众多观测到该MapPoint的特征点中挑选区分度最高的描述子
        pMP->ComputeDistinctiveDescriptors();
        // c.更新该MapPoint平均观测方向以及观测距离的范围
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        //mvIniMatches下标i表示在初始化参考帧中的特征点的序号
        //mvIniMatches[i]是初始化当前帧中的特征点的序号
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        // step 4.4：在地图中添加该MapPoint
        mpMap->AddMapPoint(pMP);
    }

    // step 4.5：更新关键帧间的连接关系
    // 在3D点和关键帧之间建立边，每个边有一个权重，边的权重是该关键帧与当前帧公共3D点的个数
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    // step 5：BA优化
    Optimizer::GlobalBundleAdjustemnt(mpMap,20);

    // step 6：将MapPoints的中值深度归一化到1，并归一化两帧之间变换
    // 评估关键帧场景深度，q=2表示中值
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f/medianDepth;
    
    //两个条件,一个是平均深度要大于0,另外一个是在当前帧中被观测到的地图点的数目应该大于100
    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    //将两帧之间的变换归一化到平均深度1的尺度下
    cv::Mat Tc2w = pKFcur->GetPose();
    // x/z y/z 将z归一化到1 
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // 把3D点的尺度也归一化到1
    //? 为什么是pKFini? 是不是就算是使用 pKFcur 得到的结果也是相同的? 
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
        }
    }

    // 这部分和SteroInitialization()相似
    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;  //也只能这样子设置了,毕竟是最近的关键帧

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;// 初始化成功，至此，初始化过程完成
}

// **************************************************************************************************************************************************
// ******************************************************** 检查上一帧中的MapPoints是否被替换 **********************************************************
// **************************************************************************************************************************************************
/*
 * @brief 检查上一帧中的MapPoints是否被替换
 * 
 * Local Mapping线程可能会将关键帧中某些MapPoints进行替换，由于tracking中需要用到mLastFrame，这里检查并更新上一帧中被替换的MapPoints
 * @see LocalMapping::SearchInNeighbors()
 */
//? 目测会发生替换,是因为合并三角化之后特别近的点吗? 
void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(pMP)//如果这个地图点存在
        {
            //获取其是否被替换,以及替换后的点
            // 这也是程序不选择将这个地图点删除的原因，因为删除了就。。。段错误了
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {   
                //然后重设一下
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}

// **************************************************************************************************************************************************
// ******************************************************** 对参考关键帧的MapPoints进行跟踪 ************************************************************
// **************************************************************************************************************************************************
/*
 * @brief 对参考关键帧的MapPoints进行跟踪
 * 
 * 1. 计算当前帧的词包，将当前帧的特征点分到特定层的nodes上
 * 2. 对属于同一node的描述子进行匹配
 * 3. 根据匹配对估计当前帧的姿态
 * 4. 根据姿态剔除误匹配
 * @return 如果匹配数大于10，返回true
 */
bool Tracking::TrackReferenceKeyFrame()
{
    // step 1：我们首先执行与参考关键帧的ORB匹配，如果找到足够的匹配，就设置一个PnP解算器
    mCurrentFrame.ComputeBoW();
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;


    int nmatches = matcher.SearchByBoW(// BoW加速匹配
        mpReferenceKF,          //参考关键帧
        mCurrentFrame,          //当前帧
        vpMapPointMatches);     //存储匹配关系

    if(nmatches<15)//匹配数超过15才继续
        return false;

    // step 2:将上一帧的位姿态作为当前帧位姿的初始值
    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    // step 3:通过优化3D-2D的重投影误差来获得位姿
    Optimizer::PoseOptimization(&mCurrentFrame);

    // step 4：剔除优化后的outlier匹配点（MapPoints）
    //之所以在优化之后才剔除外点，是因为在优化的过程中就有了对这些外点的标记
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])//如果对应到的某个特征点是外点
            {
                //清除它在当前帧中存在过的痕迹
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;//匹配的内点计数++
        }
    }

    return nmatchesMap>=10;
}

// **************************************************************************************************************************************************
// ******************************************************** 为LastFrame生成MapPoints（双目/RGB-D） ****************************************************
// **************************************************************************************************************************************************
/**
 * @brief 双目或rgbd摄像头根据深度值为最近一帧产生新的MapPoints
 *
 * 在双目和rgbd情况下，选取一些深度小一些的点（可靠一些） \n
 * 可以通过深度值产生一些新的MapPoints,用来补充当前视野中的地图点数目,这些新补充的地图点就被称之为"临时地图点""
 */
void Tracking::UpdateLastFrame()
{
    // step 1：更新最近一帧的位姿
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();
    mLastFrame.SetPose(Tlr*pRef->GetPose()); // Tlr*Trw = Tlw l:last r:reference w:world

    // 如果它是关键帧，或者单目的情况，则退出
    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR)
        return;

    // step 2：生成VO的临时MapPoints
    // 注意这些MapPoints不加入到Map中（用于计算重投影，在tracking的最后会删除）

    // step 2.1：得到最近一帧有深度值的特征点
    vector<pair<float,int> > vDepthIdx;//第一个元素是某个点的深度,第二个元素是对应的特征点id
    vDepthIdx.reserve(mLastFrame.N);

    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)

        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    //如果上一帧中没有有效深度的点,那么就直接退出了
    if(vDepthIdx.empty())
        return;

    // step 2.2：按照深度从小到大排序
    sort(vDepthIdx.begin(),vDepthIdx.end());

    // step 2.3：将距离比较近的点包装成MapPoints
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        //判断地图点是不是新加入的
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)//pMP->Observations()：获取当前地图点的被观测次数
        {
            bCreateNew = true;
        }

        //如果是一个新加入的地图点，计算新地图点的位姿（旧地图点也是从新地图点不断被观测到得来的）
        if(bCreateNew)
        {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(
                x3D,            //该点对应的空间点坐标
                mpMap,          //? 不明白为什么还要有这个参数
                &mLastFrame,    //存在这个特征点的帧(上一帧)
                i);             //特征点id

            mLastFrame.mvpMapPoints[i]=pNewMP; // 添加新的MapPoint

            // 标记为临时添加的MapPoint，之后在CreateNewKeyFrame之前会全部删除
            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else//如果不需要创建新的 临时地图点
        {
            nPoints++;
        }


        //当当前的点的深度已经超过了远点的阈值,并且已经这样处理了超过100个点的时候,说明就足够了
        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;
    }
}

// **************************************************************************************************************************************************
// ************************************************************** 由运动模型跟踪位姿 *******************************************************************
// **************************************************************************************************************************************************
/**
 * @brief 根据匀速度模型对上一帧的MapPoints进行跟踪
 * 
 * 1. 非单目情况，需要对上一帧产生一些新的MapPoints（临时） (因为传感器的原因，单目情况下仅仅凭借一帧没法生成可靠的地图点)
 * 2. 将上一帧的MapPoints投影到当前帧的图像平面上，在投影的位置进行区域匹配  NOTICE 加快了匹配的速度
 * 3. 根据匹配对估计当前帧的姿态
 * 4. 根据姿态剔除误匹配
 * @return 如果匹配数大于10，返回true
 * @see V-B Initial Pose Estimation From Previous Frame
 */
bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true);

    //为LastFrame生成MapPoints（双目/RGB-D）
    UpdateLastFrame();

    // 根据匀速运动模型得到初始位姿
    mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);
    //清空当前帧的地图点
    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    int th;//匹配过程中的搜索半径
    if(mSensor!=System::STEREO)
        th=15;
    else
        th=7;

    // step 2：对LastFrame的MapPoints进行跟踪
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);

    // 如果跟踪到的点少，则扩大搜索半径再来一次
    if(nmatches<20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR); // 2*th
    }

    //如果就算是这样还是不能够获得足够的跟踪点,那么就认为运动模型跟踪失败了
    if(nmatches<20)
        return false;

    // step 3：优化位姿
    Optimizer::PoseOptimization(&mCurrentFrame);

    // step 4：剔除外点
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                //累加成功匹配到的地图点数目
                nmatchesMap++;
        }
    }    

    if(mbOnlyTracking)
    {
        //如果在纯定位过程中追踪的地图点非常少,那么这里的 mbVO 标志就会置位
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }

    return nmatchesMap>=10;
}

// **************************************************************************************************************************************************
// ************************************************************** 对Local Map的MapPoints进行跟踪 ******************************************************
// **************************************************************************************************************************************************
/**
 * @brief 对Local Map的MapPoints进行跟踪
 * 
 * 1. 更新局部地图，包括局部关键帧和关键点
 * 2. 对局部MapPoints进行投影匹配
 * 3. 根据匹配对估计当前帧的姿态
 * 4. 根据姿态剔除误匹配
 * @return true if success
 * @see V-D track Local Map
 */
bool Tracking::TrackLocalMap()
{
    // 我们对像机的姿态和一些在帧中跟踪的地图点进行了估计，我们检索了局部地图并试图找到与局部地图中的点匹配的地方

    // step 1：更新局部关键帧mvpLocalKeyFrames和局部地图点mvpLocalMapPoints
    UpdateLocalMap();

    // step 2：在局部地图中查找与当前帧匹配的MapPoints, 其实也就是对局部地图点进行跟踪
    SearchLocalPoints();

    // step 3：更新局部所有MapPoints后对位姿再次优化
    // 在这个函数之前，在 Relocalization、TrackReferenceKeyFrame、TrackWithMotionModel 中都有位姿优化
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;

    // step 3：更新当前帧的MapPoints被观测程度，并统计跟踪局部地图的效果
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])//如果是内点
        {
            // 由于当前帧的MapPoints可以被当前帧观测到，其被观测统计量加1
            if(!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();

                if(!mbOnlyTracking)//如果不是纯定位模式
                {
                    // 该MapPoint被其它关键帧观测到过
                    //NOTICE 注意这里的"Obervation"和上面的"Found"并不是一个东西
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else//纯定位模式
                    // 记录当前帧跟踪到的MapPoints，用于统计跟踪效果
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)    //如果这个点是外点,并且当前相机输入是双目,就删除这个点
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

        }
    }

    // step 4：判断是否跟踪成功

    //如果最近刚刚发生了重定位,那么至少跟踪上了50个点我们才认为跟踪成功
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;
    //如果是正常的状态，只要跟踪的地图点大于30个我们就认为成功了
    if(mnMatchesInliers<30)
        return false;
    else
        return true;
}

// **************************************************************************************************************************************************
// ************************************************************** 判断是否将当前帧添加为关键帧***********************************************************
// **************************************************************************************************************************************************
/*
 * @brief 判断是否将当前帧添加为关键帧
 * @return true if needed
 */
bool Tracking::NeedNewKeyFrame()
{
    // step 1 如果是定位模式，就不处理
    if(mbOnlyTracking)
        return false;

    // 如果局部地图被回环程序锁定，则不插入关键帧
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpMap->KeyFramesInMap();

    // step 2：如果距离上一次插入关键帧的时间太短，就不插入
    if( mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames &&     //距离上一次重定位不超过1秒（mMaxFrames等于图像输入的帧率）
        nKFs>mMaxFrames)                                            //同时地图中的关键帧已经足够，就不插入
        return false;

    // step 3：得到参考关键帧跟踪到的MapPoints数量
	// NOTICE 在 UpdateLocalKeyFrames 函数中会将与当前帧共视程度最高的关键帧设定为参考关键帧 -- 一般的参考关键帧的选择原则
    int nMinObs = 3;//地图点的最小观测次数
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // step 4：查询局部地图管理器是否繁忙,也就是当前能否接受新的关键帧
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // step 5：对于双目或RGBD摄像头，统计 总的可以添加的MapPoints数量 和 跟踪到地图中的MapPoints数量
    int nTotal= 0;      //当前帧中可以添加到地图中的地图点数量
    int nMap = 0;       //其中可以被关键帧观测到的地图点数目
    if(mSensor!=System::MONOCULAR)// 双目或rgbd
    {
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            //如果是近点,并且这个特征点的深度合法,就可以被添加到地图中
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                nTotal++;// 当前帧中可以添加的MapPoints数
                if(mCurrentFrame.mvpMapPoints[i])
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        nMap++;// 其中可以被关键帧观测到的MapPoints数
            }
        }
    }
    else
    {
        // 单目中没有VO匹配
        nMap=1;
        nTotal=1;
    }

    //计算这个比例,当前帧中观测到的地图点数目和当前帧中总共的地图点数目之比（曾经观测点/总的观测点）
    const float ratioMap = (float)nMap/(float)(std::max(1,nTotal));

    // step 6：判断是否需要插入关键帧

    // 设定inlier阈值，和之前帧特征点匹配的inlier比例
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;// 关键帧只有一帧，那么插入关键帧的阈值设置很低
    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;//单目情况下插入关键帧的阈值很高

    // MapPoints中和地图关联的比例阈值
    float thMapRatio = 0.35f;
    if(mnMatchesInliers>300)
        thMapRatio = 0.20f;

    //插入关键帧的条件
    // 三选一
    // 1、很长时间没有插入关键帧
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // 2、localMapper处于空闲状态,才有生成关键帧的基本条件
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
    // 3、跟踪要跪的节奏
    const bool c1c =  mSensor!=System::MONOCULAR &&             //只有在双目的时候才成立
                    (mnMatchesInliers<nRefMatches*0.25 ||       //和Map点匹配的数目非常少
                      ratioMap<0.3f) ;                          //观测到KF中的点比例非常小,要挂了
    // 一个必要条件
    // 阈值比c1c要高，与之前参考帧（最近的一个关键帧）重复度不是太高
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio ||    // 还是观测到KF中的点比例较小（但比clc条件宽松）
                        ratioMap<thMapRatio) &&                     // 追踪到的Map点的比例太少,少于阈值
                    mnMatchesInliers>15);                           //匹配到的内点要足够多（否则跟踪失败了，就不用插入关键帧）

    if((c1a||c1b||c1c)&&c2)// 如果符合条件，同意插入关键帧
    {
        if(bLocalMappingIdle)
        {
            return true;
        }
        else//如果LocalMapping繁忙，就让它停止BA（BA时间长），将关键帧任务插入队列
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR)
            {
                // 队列里不能阻塞太多关键帧
                // tracking插入关键帧不是直接插入，而且先插入到mlNewKeyFrames中，
                // 然后localmapper再逐个pop出来插入到mspKeyFrames
                if(mpLocalMapper->KeyframesInQueue()<3)//队列中的关键帧数目不是很多,可以插入
                    return true;
                else
                    return false;//队列中缓冲的关键帧数目太多,暂时不能插入
            }
            else
                //对于单目情况,直接无法插入关键帧
                return false;
        }
    }
    else
        //不满足上面的条件,自然不能插入关键帧
        return false;
}

// **************************************************************************************************************************************************
// ************************************************************** 创建新的关键帧***********************************************************************
// **************************************************************************************************************************************************
/**
 * @brief 创建新的关键帧
 *
 * 对于非单目的情况，同时创建新的MapPoints
 */
void Tracking::CreateNewKeyFrame()
{
    //如果不能保持局部建图器开启的状态,就无法顺利插入关键帧
    if(!mpLocalMapper->SetNotStop(true))
        return;

    // step 1：将当前帧构造成关键帧
    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    // step 2：将当前关键帧设置为当前帧的参考关键帧
    // 在UpdateLocalKeyFrames函数中会将与当前关键帧共视程度最高的关键帧设定为当前帧的参考关键帧
    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    // 这段代码和 UpdateLastFrame 中的那一部分代码功能相同
    // step 3：对于双目或RGB-D，为当前帧生成新的MapPoints
    if(mSensor!=System::MONOCULAR)
    {
        // 根据Tcw计算mRcw、mtcw和mRwc、mOw
        mCurrentFrame.UpdatePoseMatrices();

        // step 3.1：得到当前帧中的特征点

        // 创建新的MapPoint, depth < mThDepth
        vector<pair<float,int> > vDepthIdx;//第一个元素是深度,第二个元素是对应的特征点的id
        vDepthIdx.reserve(mCurrentFrame.N);//reserve是容器预留空间，但在空间内不真正创建元素对象，mCurrentFrame.N=关键点数
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }
        // step 3.2：按照深度从小到大排序
        if(!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());

            // step 3.3：将距离比较近的点包装成MapPoints
            int nPoints = 0;//处理的近点的个数
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;
                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                if(!pMP)//如果当前帧中无这个地图点
                    bCreateNew = true;//它就是一个新的点
                else if(pMP->Observations()<1)//或者是刚刚创立
                {
                    bCreateNew = true;//它也是一个新的点
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                //如果需要新建地图点.这里是实打实的在全局地图中新建地图点
                if(bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                    // 这些添加属性的操作是每次创建MapPoint后都要做的
                    pNewMP->AddObservation(pKF,i);
                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    //不是新点，就不添加，但特征点数还是要计的
                    nPoints++;
                }

                //当当前处理的点大于深度阈值或者已经处理的点超过阈值的时候,就不再进行了
                if(vDepthIdx[j].first>mThDepth && nPoints>100)
                    break;
            }
        }
    }

    //执行插入关键帧的操作,其实是插入到队列，由mpLocalMapper来执行后面的操作
    mpLocalMapper->InsertKeyFrame(pKF);

    //然后现在允许局部建图器停止了
    mpLocalMapper->SetNotStop(false);

    //当前帧成为新的关键帧
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

// **************************************************************************************************************************************************
// ************************************************** 在LocalMap中查找在当前帧视野范围内的点 ************************************************************
// **************************************************************************************************************************************************
/**
 * @brief 对 Local MapPoints 进行跟踪
 * 
 * 在局部地图中查找在当前帧视野范围内的点，将视野范围内的点和当前帧的特征点进行投影匹配
 */
void Tracking::SearchLocalPoints()
{
    // step 1：遍历当前帧的mvpMapPoints，标记这些MapPoints不参与之后的搜索，因为当前的mvpMapPoints一定在当前帧的视野中
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())//SetBadFlag会修改mbBad标志
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                // 更新能观测到该点的帧数加1(被当前帧看到了)
                pMP->IncreaseVisible();
                // 标记该点被当前帧观测到
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                // 标记该点将来不被投影，因为已经匹配过(指的是使用恒速运动模型进行投影)
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch=0;//准备进行投影匹配的点的数目

    // step 2：将所有 LocalMapPoints 投影到当前帧，判断是否在视野范围内，然后进行投影匹配
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;

        // 已经被当前帧观测到的MapPoint不再判断是否能被当前帧观测到
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        //局部地图中的坏点也是
        if(pMP->isBad())
            continue;

        // step 2.1：判断LocalMapPoints中的点是否在在视野内
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {

            pMP->IncreaseVisible();// 观测到该点的次数+1
            nToMatch++;// 要进行投影匹配的点数+1
        }
    }

    //如果存在需要进行投影匹配的点
    if(nToMatch>0)
    {
        
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD)   //RGBD相机输入的时候,搜索的阈值会变得稍微大一些
            th=3;

        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)// 如果不久前进行过重定位，那么进行一个更加宽泛的搜索，阈值需要增大
            th=5;

        // step 2.2：对视野范围内的MapPoints通过投影进行特征点匹配
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
    }
}

// **************************************************************************************************************************************************
// ********************************************************** 更新LocalMap ***************************************************************************
// **************************************************************************************************************************************************
/**
 * @brief 更新LocalMap
 *
 * 局部地图包括：
 * - K1个关键帧、K2个临近关键帧和参考关键帧
 * - 由这些关键帧观测到的MapPoints
 */
void Tracking::UpdateLocalMap()
{
    // 供视觉使用
    // 这行程序放在UpdateLocalPoints函数后面是不是好一些
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();// 更新局部关键帧
    UpdateLocalPoints();//更新局部MapPoints
}

// **************************************************************************************************************************************************
// ********************************************************** 更新局部Points（用于UpdateLocalMap） ****************************************************
// **************************************************************************************************************************************************
/*
 * @brief 更新局部关键点，called by UpdateLocalMap()
 * 
 * 局部关键帧mvpLocalKeyFrames的MapPoints，更新mvpLocalMapPoints
 * \n 我觉得就是先把局部地图清空，然后将局部关键帧的地图点添加到局部地图中
 */
void Tracking::UpdateLocalPoints()
{
    // step 1：清空局部MapPoints
    mvpLocalMapPoints.clear();

    // step 2：遍历局部关键帧mvpLocalKeyFrames中的地图点
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        // step 2.1：将局部KF的MapPoints添加到mvpLocalMapPoints
        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;

            //如果添加过了就跳过
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)// MapPoint的属性mnTrackReferenceForFrame：防止重复添加局部MapPoint的标记
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;//设置这个标记，表示添加过了
            }
        }
    }
}

// **************************************************************************************************************************************************
// ************************************************************* 更新局部关键帧（用于UpdateLocalMap） **************************************************
// **************************************************************************************************************************************************
/**
 * @brief 更新局部关键帧，called by UpdateLocalMap()
 *
 * 遍历当前帧的MapPoints，将观测到这些MapPoints的关键帧和相邻的关键帧取出，更新 mvpLocalKeyFrames
 * //?怎么定义"相邻关键帧?" -- 从程序中来看指的就是他们的具有较好的共视关键帧,以及其父关键帧和子关键帧
 */
void Tracking::UpdateLocalKeyFrames()
{
    // step 1：遍历当前帧的MapPoints，对观测到特征点[i]的关键帧进行投票+1
    map<KeyFrame*,int> keyframeCounter;
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())
            {
                // 能观测到当前帧MapPoints的关键帧
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();//GetObservations()表示：观测到该MapPoint的KF和该MapPoint在KF中的索引
                //这里由于一个地图点可以被多个关键帧观测到,因此对于每一次观测,都获得观测到这个地图点的关键帧,并且对关键帧进行投票
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
            }
        }
    }

    if(keyframeCounter.empty())//意味着没有任何一个关键帧观测到当前的地图点
        return;

    //? 存储具有最多观测次数的关键帧?
    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    // step 2：更新局部关键帧（mvpLocalKeyFrames），添加局部关键帧有三个策略
    // 先清空局部关键帧
    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());//预先设定vector的容量

    // 策略1：能观测到当前所有点的KF都是localKF
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;
        
        //获取得票数最多的关键帧
        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;//观测到最多点的关键帧
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;// mnTrackReferenceForFrame防止重复添加局部关键帧
    }

    // 策略2：与策略1得到的局部关键帧共视程度很高的关键帧（邻居）作为局部关键帧
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        if(mvpLocalKeyFrames.size()>80)//限制数量为80
            break;

        KeyFrame* pKF = *itKF;

        // 策略2.1:最佳共视的10帧; 如果共视帧不足10帧,那么就返回所有具有共视关系的关键帧
        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);
        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                // mnTrackReferenceForFrame防止重复添加局部关键帧
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        // 策略2.2:自己的子关键帧（该帧之后的帧，有多个）
        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        // 策略2.3:自己的父关键帧（一个）
        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            // mnTrackReferenceForFrame防止重复添加局部关键帧
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }

    }

    // step 3：更新当前帧的参考关键帧（与自己共视程度最高的关键帧作为参考关键帧）
    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

// **************************************************************************************************************************************************
// ************************************************************************* 重定位过程 ***************************************************************
// **************************************************************************************************************************************************
bool Tracking::Relocalization()
{
    // step 1：计算当前帧特征点的Bow
    mCurrentFrame.ComputeBoW();

    // step 2：找到与当前帧相似的候选关键帧
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    //没有和当前帧相似的候选关键帧?完蛋,退出
    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // 我们首先执行与每个候选者的ORB匹配，如果找到足够的匹配，就设置PnP求解器
    ORBmatcher matcher(0.75,true);
    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);
    vector<vector<MapPoint*> > vvpMapPointMatches;//每个关键帧和当前帧中特征点的匹配关系
    vvpMapPointMatches.resize(nKFs);
    vector<bool> vbDiscarded;//放弃某个关键帧的标记
    vbDiscarded.resize(nKFs);

    int nCandidates=0;//有效的候选关键帧数目

    //遍历所有的候选关键帧
    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            // step 3：通过BoW进行匹配
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            //如果和当前帧的匹配点数小于15,那么只能放弃这个关键帧
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                // 初始化PnPsolver
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(
                    0.99,   //用于计算RANSAC迭代次数理论值的概率
                    10,     //最小内点数, NOTICE 但是要注意在程序中实际上是min(给定最小内点数,最小集,内点数理论值),不一定使用这个
                    300, //最大迭代次数
                    4,        //最小集(求解这个问题在一次采样中所需要采样的最少的点的个数,对于Sim3是3,EPnP是4),参与到最小内点数的确定过程中
                    0.5,     //这个是表示(最小内点数/样本总数);实际上的RANSAC正常退出的时候所需要的最小内点数其实是根据这个量来计算得到的
                    5.991);     // 目测是自由度为2的卡方检验的阈值,作为内外点判定时的距离的baseline(程序中还会根据特征点所在的图层对这个阈值进行缩放的)
                vpPnPsolvers[i] = pSolver;
                nCandidates++;//有效候选点数+1
            }
        }
    }

    // 执行P4P RANSAC，直到发现足够的 能够定位相机的内点

    bool bMatch = false;//是否已经找到相匹配的关键帧的标志
    ORBmatcher matcher2(0.9,true);

    //通过一系列骚操作,直到找到能够进行重定位的匹配上的关键帧
    while(nCandidates>0 && !bMatch)
    {
        //遍历当前所有的候选关键帧
        for(int i=0; i<nKFs; i++)
        {
            //如果刚才已经放弃了,那么这里也放弃了
            if(vbDiscarded[i])
                continue;
    
            // 执行5次RANSAC迭代
            vector<bool> vbInliers;//内点标记
            int nInliers;//内点数
            bool bNoMore;// 表示RANSAC已经没有更多的迭代次数可用 -- 也就是说数据不够好，RANSAC也已经尽力了。。。
            // step 4：通过EPnP算法估计姿态（对于每个关键帧都估计一遍）
            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // 如果这里的迭代已经尽力了
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // step 5：如果估算位姿成功，就对它进行优化
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);
                set<MapPoint*> sFound;//成功被再次找到的 地图点的集合,其实就是经过RANSAC之后的内点

                const int np = vbInliers.size();
                //遍历所有内点
                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                // 开始优化
                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);//只优化位姿,不优化地图点的坐标;返回的是内点的数量

                //如果优化之后的内点数目不多,注意这里是直接跳过了本次循环,但是却没有放弃当前的这个关键帧
                if(nGood<10)
                    continue;
                //删除外点对应的地图点
                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // step 6：如果内点较少，则通过投影的方式对之前未匹配的点进行匹配，再进行优化求解
                // 前面的匹配关系是用词袋匹配过程得到的
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(
                        mCurrentFrame,      //当前帧
                        vpCandidateKFs[i],      //关键帧
                        sFound,                 //已经找到的地图点集合
                        10,                 //窗口阈值
                        100);           //ORB描述子距离

                    //如果通过投影过程match了比较多的特征点，就再次进行优化
                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        //如果这样内点数还是比较少的话,就使用更小的窗口搜索投影点;由于相机位姿已经使用了更多的点进行了优化,所以可以认为使用更小的窗口搜索能够取得意料之内的效果
                        if(nGood>30 && nGood<50)
                        {
                            //重新进行搜索
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(
                                mCurrentFrame,      //当前帧
                                vpCandidateKFs[i],      //候选的关键帧
                                sFound,                 //已经找到的地图点
                                3,                  //新的窗口阈值
                                64);            //ORB距离?


                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);// 最后优化
                                //更新地图点
                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }
                //如果对于当前的关键帧已经有足够的内点(50个)了,那么就认为当前的这个关键帧已经和当前帧匹配上了
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }//遍历所有的候选关键帧
            // ? 大哥，这里PnPSolver 可不能够保证一定能够得到相机位姿啊？怎么办？
        }//一直运行,直到已经没有足够的关键帧,或者是已经有成功匹配上的关键帧
    }

    //折腾了这么久还是没有匹配上
    if(!bMatch)
    {
        return false;
    }
    else
    {
        //如果匹配上了,说明当前帧重定位成功了(也就是在上面的优化过程中,当前帧已经拿到了属于自己的位姿).因此记录当前帧的id
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }
}//重定位

// **************************************************************************************************************************************************
// ********************************************************************* 复位Tracking线程 ************************************************************
// **************************************************************************************************************************************************
void Tracking::Reset()
{
    //基本上是挨个请求各个线程终止
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }
    cout << "System Reseting" << endl;

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    //然后复位各种变量
    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if(mpViewer)
        mpViewer->Release();
}

// **************************************************************************************************************************************************
// ********************************************************************* 从文件读取配置信息 ************************************************************
// **************************************************************************************************************************************************
void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    //做标记,表示在初始化帧的时候将会是第一个帧,要对它进行一些特殊的初始化操作
    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

}
