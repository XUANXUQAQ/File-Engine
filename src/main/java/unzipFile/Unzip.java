package unzipFile;

import java.io.*;
import java.nio.charset.Charset;
import java.util.Enumeration;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

public class Unzip {
    private static Unzip unzip = new Unzip();

    public static Unzip getInstance() {
        return unzip;
    }

    private Unzip() {
    }

    public void unZipFiles(File zipFile, String descDir) {

        ZipFile zip = null;//解决中文文件夹乱码
        try {
            zip = new ZipFile(zipFile, Charset.forName("GBK"));
        } catch (IOException ignored) {

        }

        assert zip != null;
        for (Enumeration<? extends ZipEntry> entries = zip.entries(); entries.hasMoreElements(); ) {
            ZipEntry entry = entries.nextElement();
            String zipEntryName = entry.getName();
            try (InputStream in = zip.getInputStream(entry)) {
                String outPath = (descDir + "/" + zipEntryName).replaceAll("\\*", "/");
                // 判断路径是否存在,不存在则创建文件路径
                File file = new File(outPath.substring(0, outPath.lastIndexOf('/')));
                if (!file.exists()) {
                    file.mkdirs();
                }
                // 判断文件全路径是否为文件夹,如果是上面已经上传,不需要解压
                if (new File(outPath).isDirectory()) {
                    continue;
                }
                try (FileOutputStream out = new FileOutputStream(outPath)) {
                    byte[] buf1 = new byte[1024];
                    int len;
                    while ((len = in.read(buf1)) > 0) {
                        out.write(buf1, 0, len);
                    }
                } catch (FileNotFoundException ignored) {

                }
            } catch (IOException ignored) {

            }
        }
    }
}
