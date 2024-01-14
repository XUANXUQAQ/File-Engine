package file.engine.configs.core;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class CoreAdvancedConfigEntity {

    private long searchWarmupTimeoutInMills;

    private long waitForSearchTasksTimeoutInMills;

    private boolean isDeleteUsnOnExit;

    private long restartMonitorDiskThreadTimeoutInMills;
}
