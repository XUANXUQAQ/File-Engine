package FileEngine.frames;

import FileEngine.configs.Enums;
import FileEngine.eventHandler.EventManagement;
import FileEngine.utils.TranslateUtil;
import FileEngine.utils.download.DownloadManager;
import FileEngine.utils.download.DownloadUtil;

import javax.swing.*;
import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * 使用方法：给点击下载的button添加actionListener，用一个AtomicBoolean isDownloadStarted判断当前按钮是点击下载，还是点击取消下载
 * 如果是点击取消下载（isDownloadStarted = true）则发送一个cancel请求
 * 如果是点击下载，则通过判断后发送一个start下载请求，然后开启一个线程运行该方法即可
 *
 * 不要自己设置isDownloadStarted
 */
public class SetDownloadProgress {

    /**
     * 当你点击下载按钮时使用，此时isDownloadStarted必须设为true
     * @param labelProgress 显示进度的label
     * @param buttonInstall 设置文字为下载还是取消的下载点击按钮
     * @param downloadManager 下载管理类的实例
     * @param isDownloadStarted 必须为true，在下载失败后会被恢复为false，用于在buttonInstall判断点击时是取消下载还是开始下载
     * @param successSign 下载成功后创建文件
     * @param currentTaskStr 当前任务的标志
     * @param getSelectedMethod 线程需要从哪个方法获取字符串，当获取的字符串不等于currentTaskStr时，则会停止设置buttonInstall和labelProgress的值
     * @param invokeMethodObj 执行method需要的实例
     */
    protected static void setProgress(JLabel labelProgress,
                                   JButton buttonInstall,
                                   DownloadManager downloadManager,
                                   AtomicBoolean isDownloadStarted,
                                   File successSign,
                                   String currentTaskStr,
                                   Method getSelectedMethod,
                                   Object invokeMethodObj) {
        try {
            isDownloadStarted.set(true);
            TranslateUtil translateUtil = TranslateUtil.getInstance();
            DownloadUtil downloadUtil = DownloadUtil.getInstance();
            EventManagement eventManagement = EventManagement.getInstance();
            String buttonOriginalText = buttonInstall.getText();
            boolean isStarted = true;
            boolean isDownloadStartedSet = false;    //当getSelectedMethod获取到的字符串不等于currentTaskStr，让isDownloadStarted设置为false（只设置一次，之后会由其他线程托管）
            while (isStarted) {
                if (!eventManagement.isNotMainExit()) {
                    return;
                }
                String taskStrFromMethod = currentTaskStr;
                if (getSelectedMethod != null) {
                    taskStrFromMethod = (String) getSelectedMethod.invoke(invokeMethodObj);
                }
                if (currentTaskStr.equals(taskStrFromMethod)) {
                    isDownloadStarted.set(true);
                    isDownloadStartedSet = false;
                    double progress = downloadUtil.getDownloadProgress(downloadManager);
                    Enums.DownloadStatus downloadStatus = downloadUtil.getDownloadStatus(downloadManager);
                    if (downloadStatus == Enums.DownloadStatus.DOWNLOAD_DONE) {
                        //下载完成，禁用按钮
                        labelProgress.setText("");
                        buttonInstall.setText(translateUtil.getTranslation("Downloaded"));
                        buttonInstall.setEnabled(false);
                        isDownloadStarted.set(false);
                        isStarted = false;
                        if (!successSign.exists()) {
                            if (!successSign.createNewFile()) {
                                throw new RuntimeException("创建更新标识符失败");
                            }
                        }
                    } else if (downloadStatus == Enums.DownloadStatus.DOWNLOAD_ERROR) {
                        //下载错误，重置button
                        labelProgress.setText(translateUtil.getTranslation("Download failed"));
                        buttonInstall.setText(buttonOriginalText);
                        buttonInstall.setEnabled(true);
                        isDownloadStarted.set(false);
                        isStarted = false;
                    } else if (downloadStatus == Enums.DownloadStatus.DOWNLOAD_DOWNLOADING) {
                        //正在下载
                        labelProgress.setText(translateUtil.getTranslation("Downloading:") + (int) (progress * 100) + "%");
                        buttonInstall.setText(translateUtil.getTranslation("Cancel"));
                        buttonInstall.setEnabled(true);
                    } else if (downloadStatus == Enums.DownloadStatus.DOWNLOAD_INTERRUPTED) {
                        //用户自行中断
                        labelProgress.setText("");
                        buttonInstall.setText(buttonOriginalText);
                        buttonInstall.setEnabled(true);
                        isDownloadStarted.set(false);
                        isStarted = false;
                    }
                } else {
                    if (!isDownloadStartedSet) {
                        isDownloadStartedSet = true;
                        isDownloadStarted.set(false);
                    }
                }
                TimeUnit.MILLISECONDS.sleep(50);
            }
        } catch (InterruptedException | IOException | InvocationTargetException | IllegalAccessException e) {
            e.printStackTrace();
        }
    }
}
