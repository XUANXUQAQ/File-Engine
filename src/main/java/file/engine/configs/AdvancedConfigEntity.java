package file.engine.configs;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class AdvancedConfigEntity {

    private long searchWarmupTimeoutInMills;

    private long waitForInputAndPrepareSearchTimeoutInMills;

    private long waitForInputAndStartSearchTimeoutInMills;

    private long waitForSearchTasksTimeoutInMills;

    private long clearIconCacheTimeoutInMills;

    private boolean isDeleteUsnOnExit;

    private long restartMonitorDiskThreadTimeoutInMills;
}
