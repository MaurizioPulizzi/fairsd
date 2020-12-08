"""
Access and store files when needed.

Server with files
=================

Server provides files through HTTP. Any HTTP server that can serve
static files can work, including Apache, Nginx and Python's HTTP server.

Files can be organized in subfolders. Each file can have a
corresponding info file (with .info extension).

A test server could be made by just creating a new empty folder and
creating a subfolder "additional-data" there with the following files::

  additional-data/a-very-big-file.txt
  additional-data/a-very-big-file.txt.info

Our .info file should contain the following::

  {"tags": [ "huge file", "example" ], "datetime": "2016-10-10 11:39:07"}

Then we can start a test server with::

  python -m http.server

To access the server and download the file we could use::

  >>> import serverfiles
  >>> sf = serverfiles.ServerFiles(server="http://localhost:8000/")
  >>> sf.listfiles()
  [('additional-data', 'a-very-big-file.txt')]
  >>> lf = serverfiles.LocalFiles("sftest", serverfiles=sf)
  >>> lf.download('additional-data', 'a-very-big-file.txt')


Info files
===========

Info files, which have an additional .info extension,
must be SON dictionaries. Keys that are read by this module are:

* datetime ("%Y-%m-%d %H:%M:%S"),

* compression (if set, the file is uncompressed automatically,
  can be one of .bz2, .gz, .tar.gz, .tar.bz2),

* and tags (a list of strings).

Server query optimization
=========================

A server can contain a __INFO__ file in its root folder. This file is
a JSON list, whose elements are lists of [ list-of-path, info dictionary ].
If such file exists its contents will be used instead of server queries
for file listing and info lookup, which is critical for high latency
connections. Such file can be prepared as:

>>> sf = ServerFiles(server="yourserver")
>>> json.dump(list(sf.allinfo().items()), open("__INFO__", "wt"))

If your server already has an __INFO__ file, the above code will just get
its contents.


Remote files
============

.. autoclass:: ServerFiles
    :members:


Local files
===========

.. autoclass:: LocalFiles
    :members:

"""

import functools
try:
    import urllib.parse as urlparse
except ImportError:
    import urlparse
from contextlib import contextmanager
import threading
import os
import tarfile
import gzip
import bz2
import datetime
import tempfile
import json
try:
    from html.parser import HTMLParser
except ImportError:
    from HTMLParser import HTMLParser
import shutil

import requests
import requests.exceptions

try:
    FileNotFoundError
except:
    FileNotFoundError = IOError


# default socket timeout in seconds
TIMEOUT = 5


def _open_file_info(fname):
    with open(fname, 'rt') as f:
        return json.load(f)


def _save_file_info(fname, info):
    with open(fname, 'wt') as f:
        json.dump(info, f)


def _create_path(target):
    try:
        os.makedirs(target)
    except OSError:
        pass


def _is_prefix(pref, whole):
    if len(pref) > len(whole):
        return False
    for a, b in zip(pref, whole):
        if a != b:
            return False
    return True


class _FindLinksParser(HTMLParser, object):

    def __init__(self):
        super(_FindLinksParser, self).__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for name, value in attrs:
                if name == "href":
                    #ignore navidation and hidden files
                    if value.startswith("?") or value.startswith("/") or \
                       value.startswith(".") or value.startswith("__"):
                        continue
                    self.links.append(urlparse.unquote(value))


class ServerFiles:
    """A class for listing or downloading files from the server."""

    def __init__(self, server, username=None, password=None):
        if server.endswith('/'):
            self.server = server
        else:
            self.server = server + '/'
        """Server URL."""
        self.username = username
        """Username for authenticated HTTP queried."""
        self.password = password
        """Password for authenticated HTTP queried."""

        self.req = requests.Session()
        a = requests.adapters.HTTPAdapter(max_retries=2)
        self.req.mount('https://', a)
        self.req.mount('http://', a)

        # cached info for all files on server
        # None is not loaded, False if it does not exist
        self._info = None

    def _download_server_info(self):
        if self._info is None:
            t = self._open("__INFO__")
            if t.status_code == 200:
                self._info = {tuple(a): b for a, b in json.loads(t.text)}
            else:
                self._info = False #do not check again

    def listfiles(self, *args, **kwargs):
        """Return a list of files on the server. Do not list .info files."""
        recursive = kwargs.get("recursive", True)
        self._download_server_info()
        if self._info:
            return [a for a in self._info.keys() if _is_prefix(args, a)]
        text = self._open(*args).text
        parser = _FindLinksParser()
        parser.feed(text)
        links = parser.links
        files = [args + (f,) for f in links if not f.endswith("/") and not f.endswith(".info")]
        if recursive:
            for f in links:
                if f.endswith("/"):
                    f = f.strip("/")
                    nargs = args + (f,)
                    files.extend([a for a in self.listfiles(*nargs, recursive=True)])
        return files

    def download(self, *path, **kwargs):
        """
        Download a file and name it with target name. Callback
        is called once for each downloaded percentage.
        """
        callback = kwargs.get("callback", None)
        target = kwargs.get("target", None)
        _create_path(os.path.dirname(target))

        req = self._open(*path)
        if req.status_code == 404:
            raise FileNotFoundError
        elif req.status_code != 200:
            raise IOError

        size = req.headers.get('content-length')
        if size:
            size = int(size)

        f = tempfile.TemporaryFile()

        chunksize = 1024*8
        lastchunkreport= 0.0001

        readb = 0

        for buf in req.iter_content(chunksize):
            readb += len(buf)
            while size and float(readb) / size > lastchunkreport+0.01:
                lastchunkreport += 0.01
                if callback:
                    callback()
            f.write(buf)

        f.seek(0)

        with open(target, "wb") as fo:
            shutil.copyfileobj(f, fo)

        if callback and not size: #size was unknown, call callbacks
            for i in range(99):
                callback()

        if callback:
            callback()

    def allinfo(self, *path, **kwargs):
        """Return all info files in a dictionary, where keys are paths."""
        recursive = kwargs.get("recursive", True)
        self._download_server_info()
        files = self.listfiles(*path, recursive=recursive)
        infos = {}
        for npath in files:
            infos[npath] = self.info(*npath)
        return infos

    def search(self, sstrings, **kwargs):
        """
        Search for files on the repository where all substrings in a list
        are contained in at least one choosen field (tag, title, name). Return
        a list of tuples: first tuple element is the file's domain, second its
        name. As for now the search is performed locally, therefore
        information on files in repository is transfered on first call of
        this function.
        """
        if self._info is None or self._info is False:
            self._info = self.allinfo()
        return _search(self._info, sstrings, **kwargs)

    def info(self, *path):
        """Return a dictionary containing repository file info."""
        self._download_server_info()
        if self._info:
            return self._info.get(path, {})
        path = list(path)
        path[-1] += ".info"
        t = self._open(*path)
        if t.status_code == 200:
            return json.loads(t.text)
        else:
            return {}

    def _server_request(self, root, *path):
        auth = None
        if self.username and self.password:
            auth = (self.username, self.password)
        return self.req.get(root + "/".join(path), auth=auth,
                            timeout=TIMEOUT, stream=True)

    def _open(self, *args):
        return self._server_request(self.server, *args)


def _keyed_lock(lock_constructor=threading.Lock):
    lock = threading.Lock()
    locks = {}
    def get_lock(key):
        with lock:
            if key not in locks:
                locks[key] = lock_constructor()
            return locks[key]
    return get_lock


#using RLock instead of Ales's Orange 2 solution
_get_lock = _keyed_lock(threading.RLock)


def _split_path(head):
    out = []
    while True:
        head, tail = os.path.split(head)
        out.insert(0, tail)
        if not head:
            break
    return out


class LocalFiles:
    """Manage local files."""

    def __init__(self, path, serverfiles=None):
        self.serverfiles_dir = path
        """A folder downloaded files are stored in."""
        _create_path(self.serverfiles_dir)
        self.serverfiles = serverfiles
        """A ServerFiles instance."""

    @contextmanager
    def _lock_file(self, *args):
        path = self.localpath(*args)
        path = os.path.normpath(os.path.realpath(path))
        lock = _get_lock(path)
        lock.acquire(True)
        try:
            yield
        finally:
            lock.release()

    def _locked(f):
        @functools.wraps(f)
        def func(self, *path, **kwargs):
            with self._lock_file(*path):
                return f(self, *path, **kwargs)
        func.unwrapped = f
        return func

    def localpath(self, *args):
        """ Return the local location for a file. """
        return os.path.join(os.path.expanduser(self.serverfiles_dir), *args)

    @_locked
    def download(self, *path, **kwargs):
        """Download file from the repository. Callback can be a function without
        arguments and will be called once for each downloaded percent of
        file: 100 times for the whole file. If extract is True, files
        marked as compressed will be uncompressed after download."""
        extract = kwargs.get("extract", True)
        callback = kwargs.get("callback", None)
        info = self.serverfiles.info(*path)

        extract = extract and "compression" in info
        target = self.localpath(*path)
        self.serverfiles.download(*path,
                                  target=target + ".tmp" if extract else target,
                                  callback=callback)

        _save_file_info(target + '.info', info)

        if extract:
            if info.get("compression") in ["tar.gz", "tar.bz2"]:
                f = tarfile.open(target + ".tmp")
                try:
                    os.mkdir(target)
                except OSError:
                    pass
                f.extractall(target)
            elif info.get("compression") == "gz":
                f = gzip.open(target + ".tmp")
                shutil.copyfileobj(f, open(target, "wb"))
            elif info.get("compression") == "bz2":
                f = bz2.BZ2File(target + ".tmp", "r")
                shutil.copyfileobj(f, open(target, "wb"))
            f.close()
            os.remove(target + ".tmp")

    @_locked
    def localpath_download(self, *path, **kwargs):
        """
        Return local path for the given domain and file. If file does not exist,
        download it. Additional arguments are passed to the :obj:`download` function.
        """
        pathname = self.localpath(*path)
        if not os.path.exists(pathname):
            self.download.unwrapped(self, *path, **kwargs)
        return pathname

    def listfiles(self, *path):
        """List files (or folders) in local repository that have
        corresponding .info files.  Do not list .info files."""
        dir = self.localpath(*path)
        files = []
        for root, dirs, fnms in os.walk(dir):
            for f in fnms:
                if f[-5:] == '.info' and os.path.exists(os.path.join(root, f[:-5])):
                    try:
                        _open_file_info(os.path.join(root, f))
                        files.append(
                            path + tuple(_split_path(
                                os.path.relpath(os.path.join(root, f[:-5]), start=dir)
                            )))
                    except ValueError:
                        pass
        return files

    def info(self, *path):
        """Return .info file for a file in a local repository."""
        target = self.localpath(*path)
        return _open_file_info(target + '.info')

    def allinfo(self, *path):
        """Return all local info files in a dictionary, where keys are paths."""
        files = self.listfiles(*path)
        dic = {}
        for filename in files:
            dic[filename] = self.info(*filename)
        return dic

    def needs_update(self, *path):
        """Return True if a file does not exist in the local repository,
        if there is a newer version on the server or if either
        version can not be determined."""
        dt_fmt = "%Y-%m-%d %H:%M:%S"
        try:
            linfo = self.info(*path)
            dt_local = datetime.datetime.strptime(
                            linfo["datetime"][:19], dt_fmt)
            dt_server = datetime.datetime.strptime(
                self.serverfiles.info(*path)["datetime"][:19], dt_fmt)
            return dt_server > dt_local
        except FileNotFoundError:
            return True
        except KeyError:
            return True

    def update(self, *path, **kwargs):
        """Download the corresponding file from the server if server
        copy was updated.
        """
        if self.needs_update(*path):
            self.download(*path, **kwargs)

    def search(self, sstrings, **kwargs):
        """Search for files in the local repository where all substrings in a list
        are contained in at least one chosen field (tag, title, name). Return a
        list of tuples: first tuple element is the domain of the file, second
        its name."""
        si = self.allinfo()
        return _search(si, sstrings, **kwargs)

    def update_all(self, *path):
        for fu in self.listfiles(*path):
            self.update(*fu)

    @_locked
    def remove(self, *path):
        """Remove a file of a path from local repository."""
        path = self.localpath(*path)
        if os.path.exists(path + ".info"):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                elif os.path.isfile(path):
                    os.remove(path)
                os.remove(path + ".info")
            except OSError as ex:
                print("Failed to delete", path, "due to:", ex)
        else:
            raise FileNotFoundError


def _search(si, sstrings, case_sensitive=False, in_tag=True, in_title=True, in_name=True):
    found = []

    for path, info in si.items():
        target = ""
        if in_tag: target += " ".join(info.get('tags', []))
        if in_title: target += info.get('title', "")
        if in_name: target += " ".join(path)
        if not case_sensitive: target = target.lower()

        match = True
        for s in sstrings:
            if not case_sensitive:
                s = s.lower()
            if s not in target:
                match = False
                break

        if match:
            found.append(path)

    return found


def sizeformat(size):
    """
    >>> sizeformat(256)
    '256 bytes'
    >>> sizeformat(1024)
    '1.0 KB'
    >>> sizeformat(1.5 * 2 ** 20)
    '1.5 MB'

    """
    for unit in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            if unit == "bytes":
                return "%1.0f %s" % (size, unit)
            else:
                return "%3.1f %s" % (size, unit)
        size /= 1024.0
    return "%.1f PB" % size


if __name__ == '__main__':
    sf = ServerFiles()
    lf = LocalFiles()
    info = sf.allinfo()
    print(os.getcwd())
    with open("__INFO__.json", "wt") as fo:
        json.dump(list(info.items()), fo)
