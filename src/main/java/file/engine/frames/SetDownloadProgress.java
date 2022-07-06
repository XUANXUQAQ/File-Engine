package file.engine.frames;

import file.engine.configs.Constants;
import file.engine.event.handler.EventManagement;
import file.engine.services.TranslateService;
import file.engine.services.download.DownloadManager;

import javax.swing.*;
import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;

/**
 * 使用方法：给点击下载的button添加actionListener，判断当前按钮是点击下载，还是点击取消下载
 * 如果是点击取消下载则发送一个cancel请求
 * 如果是点击下载，则通过判断后发送一个start下载请求，然后开启一个线程运行该方法即可
 */
public class SetDownloadProgress {

    /**
     * 当你点击下载按钮时使用，此时isDownloadStarted必须设为true
     *
     * @param labelProgress           显示进度的label
     * @param buttonInstall           设置文字为下载还是取消的下载点击按钮
     * @param downloadManager         下载管理类的实例
     * @param isLabelAndButtonVisible 判断是否需要显示label和button的状态
     * @param successSign             下载成功后创建文件
     * @param currentTaskStr          当前任务的标志
     * @param getSelectedMethod       线程需要从哪个方法获取任务的标志，当获取的字符串不等于currentTaskStr时，则会停止设置buttonInstall和labelProgress的值
     * @param invokeMethodObj         执行method需要的实例
     */
    protected static boolean setProgress(JLabel labelProgress,
                                         JButton buttonInstall,
                                         DownloadManager downloadManager,
                                         Supplier<Boolean> isLabelAndButtonVisible,
                                         File successSign,
                                         String currentTaskStr,
                                         Method getSelectedMethod,
                                         Object invokeMethodObj) {
        boolean retVal = false;
        try {
            TranslateService translateService = TranslateService.getInstance();
            EventManagement eventManagement = EventManagement.getInstance();
            String buttonOriginalText = buttonInstall.getText();
            boolean isStarted = true;
            while (isStarted) {
                if (isLabelAndButtonVisible.get()) {
                    if (!eventManagement.notMainExit()) {
                        return true;
                    }
                    String taskStrFromMethod = currentTaskStr;
                    if (getSelectedMethod != null) {
                        taskStrFromMethod = (String) getSelectedMethod.invoke(invokeMethodObj);
                    }
                    if (currentTaskStr.equals(taskStrFromMethod)) {
                        double progress = downloadManager.getDownloadProgress();
                        Constants.Enums.DownloadStatus downloadStatus = downloadManager.getDownloadStatus();
                        if (downloadStatus == Constants.Enums.DownloadStatus.DOWNLOAD_DONE) {
                            //下载完成，禁用按钮
                            labelProgress.setText("");
                            buttonInstall.setText(translateService.getTranslation("Downloaded"));
                            buttonInstall.setEnabled(false);
                            isStarted = false;
                            if (!successSign.exists()) {
                                if (!successSign.createNewFile()) {
                                    throw new RuntimeException("创建更新标识符失败");
                                }
                            }
                            retVal = true;
                        } else if (downloadStatus == Constants.Enums.DownloadStatus.DOWNLOAD_ERROR) {
                            //下载错误，重置button
                            labelProgress.setText(translateService.getTranslation("Download failed"));
                            buttonInstall.setText(buttonOriginalText);
                            buttonInstall.setEnabled(true);
                            isStarted = false;
                        } else if (downloadStatus == Constants.Enums.DownloadStatus.DOWNLOAD_DOWNLOADING) {
                            //正在下载
                            labelProgress.setText(translateService.getTranslation("Downloading:") + (int) (progress * 100) + "%");
                            buttonInstall.setText(translateService.getTranslation("Cancel"));
                            buttonInstall.setEnabled(true);
                        } else if (downloadStatus == Constants.Enums.DownloadStatus.DOWNLOAD_INTERRUPTED) {
                            //用户自行中断
                            labelProgress.setText("");
                            buttonInstall.setText(buttonOriginalText);
                            buttonInstall.setEnabled(true);
                            isStarted = false;
                            retVal = true;
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
