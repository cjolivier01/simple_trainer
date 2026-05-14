import errno
import os
import pty

import pytest

from tools.trainer import close_inherited_pty_masters


def test_close_inherited_pty_masters_closes_extra_ptmx_fd() -> None:
    master_fd, slave_fd = pty.openpty()
    try:
        assert os.readlink(f"/proc/self/fd/{master_fd}") in {
            "/dev/ptmx",
            "/dev/pts/ptmx",
        }

        close_inherited_pty_masters()

        with pytest.raises(OSError) as exc_info:
            os.fstat(master_fd)
        assert exc_info.value.errno == errno.EBADF
        os.fstat(slave_fd)
    finally:
        for fd_num in (master_fd, slave_fd):
            try:
                os.close(fd_num)
            except OSError:
                pass
