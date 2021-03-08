package FileEngine.frames;

import FileEngine.configs.Enums;
import FileEngine.eventHandler.EventManagement;
import FileEngine.services.download.DownloadManager;
import FileEngine.services.download.DownloadService;
import FileEngine.utils.TranslateUtil;

import javax.swing.*;
import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;

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
     * @param func 判断是否需要显示label和button的状态
     * @param successSign 下载成功后创建文件
     * @param currentTaskStr 当前任务的标志
     * @param getSelectedMethod 线程需要从哪个方法获取字符串，当获取的字符串不等于currentTaskStr时，则会停止设置buttonInstall和labelProgress的值
     * @param invokeMethodObj 执行method需要的实例
     */
    protected static boolean setProgress(JLabel labelProgress,
                                   JButton buttonInstall,
                                   DownloadManager downloadManager,
                                   Supplier<Boolean> func,
                                   File successSign,
                                   String currentTaskStr,
                                   Method getSelectedMethod,
                                   Object invokeMethodObj) {
        boolean retVal = false;
        try {
            TranslateUtil translateUtil = TranslateUtil.getInstance();
            DownloadService downloadService = DownloadService.getInstance();
            EventManagement eventManagement = EventManagement.getInstance();
            String buttonOriginalText = buttonInstall.getText();
            boolean isStarted = true;
            boolean isDownloadStartedSet = false;    //当getSelectedMethod获取到的字符串不等于currentTaskStr，让isDownloadStarted设置为false（只设置一次，之后会由其他线程托管）
            while (isStarted) {
                if (func.get()) {
                    if (!eventManagement.isNotMainExit()) {
                        return true;
                    }
                    String taskStrFromMethod = currentTaskStr;
                    if (getSelectedMethod != null) {
                        taskStrFromMethod = (String) getSelectedMethod.invoke(invokeMethodObj);
                    }
                    if (currentTaskStr.equals(taskStrFromMethod)) {
                        isDownloadStartedSet = false;
                        double progress = downloadService.getDownloadProgress(downloadManager);
                        Enums.DownloadStatus downloadStatus = downloadService.getDownloadStatus(downloadManager);
                        if (downloadStatus == Enums.DownloadStatus.DOWNLOAD_DONE) {
                            //下载完成，禁用按钮
                            labelProgress.setText("");
                            buttonInstall.setText(translateUtil.getTranslation("Downloaded"));
                            buttonInstall.setEnabled(false);
                            isStarted = false;
                            if (!successSign.exists()) {
                                if (!successSign.createNewFile()) {
                                    throw new RuntimeException("创建更新标识符失败");
                                }
                            }
                            retVal = true;
                        } else if (downloadStatus == Enums.DownloadStatus.DOWNLOAD_ERROR) {
                            //下载错误，重置button
                            labelProgress.setText(translateUtil.getTranslation("Download failed"));
                            buttonInstall.setText(buttonOriginalText);
                            buttonInstall.setEnabled(true);
                            isStarted = false;
                            retVal = false;
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
                            isStarted = false;
                            retVal = true;
                        }
                    } else {
                        if (!isDownloadStartedSet) {
                            isDownloadStartedSet = true;
                        }
                    }
                }
                TimeUnit.MILLISECONDS.sleep(50);
            }
        } catch (InterruptedException | IOException | InvocationTargetException | IllegalAccessException e) {
            e.printStackTrace();
        }
        return retVal;
    }
}
