package file.engine.utils;

import oshi.SystemInfo;
import oshi.hardware.GlobalMemory;
import oshi.hardware.HardwareAbstractionLayer;
import oshi.util.GlobalConfig;

public class SystemInfoUtil {
    private static final SystemInfo si = new SystemInfo();
    private static final HardwareAbstractionLayer hal = si.getHardware();
    private static final GlobalMemory memory = hal.getMemory();

    static {
        GlobalConfig.set(GlobalConfig.OSHI_OS_WINDOWS_PERFOS_DIABLED, false);
        GlobalConfig.set(GlobalConfig.OSHI_OS_WINDOWS_PERFPROC_DIABLED, false);
        GlobalConfig.set(GlobalConfig.OSHI_OS_WINDOWS_PERFDISK_DIABLED, false);
    }

    public static double getMemoryUsage() {

        long totalByte = memory.getTotal();
        long availableMemory = memory.getAvailable();
        return (totalByte - availableMemory) * 1.0 / totalByte;
    }
}
