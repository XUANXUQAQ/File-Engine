package file.engine.utils;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.compress.archivers.sevenz.SevenZArchiveEntry;
import org.apache.commons.compress.archivers.sevenz.SevenZFile;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

@Slf4j
public class CompressUtil {
    /**
     * 解压缩7z文件
     *
     * @param file       压缩包文件
     * @param targetPath 目标文件夹
     */
    public static void decompress7Z(File file, String targetPath) {
        try (var sevenZFile = new SevenZFile(file)) {
            // 创建输出目录
            createDirectory(targetPath, null);
            SevenZArchiveEntry entry;

            while ((entry = sevenZFile.getNextEntry()) != null) {
                if (entry.isDirectory()) {
                    createDirectory(targetPath, entry.getName()); // 创建子目录
                } else {
                    try (var outputStream = new FileOutputStream(targetPath + File.separator + entry.getName())) {
                        int len;
                        byte[] b = new byte[2048];
                        while ((len = sevenZFile.read(b)) != -1) {
                            outputStream.write(b, 0, len);
                        }
                        outputStream.flush();
                    }
                }
            }
        } catch (IOException e) {
            log.error(e.getMessage(), e);
        }
    }

    /**
     * 构建目录
     *
     * @param outputDir 输出目录
     * @param subDir    子目录
     */
    private static void createDirectory(String outputDir, String subDir) {
        File file = new File(outputDir);
        if (!(subDir == null || subDir.trim().isEmpty())) {//子目录不为空
            file = new File(outputDir + File.separator + subDir);
        }
        if (!file.exists()) {
            if (!file.getParentFile().exists()) {
                file.getParentFile().mkdirs();
            }
            file.mkdirs();
        }
    }
}
