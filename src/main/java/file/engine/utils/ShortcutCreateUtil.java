package file.engine.utils;

import file.engine.event.handler.EventManagement;
import file.engine.event.handler.impl.taskbar.ShowTaskBarMessageEvent;
import file.engine.services.TranslateService;
import file.engine.utils.file.FileUtil;

import java.io.File;
import java.io.IOException;

public class ShortcutCreateUtil {

    /**
     * 创建需要打开的文件的快捷方式
     *
     * @param fileOrFolderPath  文件路径
     * @param writeShortCutPath 保存快捷方式的位置
     * @throws IOException 创建错误
     */
    public static void createShortCut(String fileOrFolderPath, String writeShortCutPath, boolean isNotifyUser) throws IOException {
        EventManagement eventManagement = EventManagement.getInstance();
        TranslateService translateService = TranslateService.getInstance();
        String lower = fileOrFolderPath.toLowerCase();
        if (lower.endsWith(".lnk") || lower.endsWith(".url")) {
            //直接复制文件
            FileUtil.copyFile(new File(fileOrFolderPath), new File(writeShortCutPath));
        } else {
            File shortcutGen = new File("user/shortcutGenerator.vbs");
            String shortcutGenPath = shortcutGen.getAbsolutePath();
            String start = shortcutGenPath.substring(0, 2);
            String end = "\"" + shortcutGenPath.substring(2) + "\"";
            String workingDir = fileOrFolderPath.substring(0, fileOrFolderPath.lastIndexOf(File.separator));
            String commandToGenLnk = start + end + " /target:" + fileOrFolderPath.substring(0, 2) + "\"" + fileOrFolderPath.substring(2) + "\"" + " " +
                    "/shortcut:" + writeShortCutPath.substring(0, 2) + "\"" + writeShortCutPath.substring(2) + "\"" + " " +
                    "/workingdir:" + workingDir.substring(0, 2) + "\"" + workingDir.substring(2) + "\"";
            Runtime.getRuntime().exec(new String[]{"cmd.exe", "/c", commandToGenLnk});
        }
        if (isNotifyUser) {
            eventManagement.putEvent(new ShowTaskBarMessageEvent(
                    translateService.getTranslation("Info"),
                    translateService.getTranslation("Shortcut created")));
        }
    }
}
