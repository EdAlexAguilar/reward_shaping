token=`cat .token`
psw=`cat .password`
logdir=$1

echo -e "[Info] Creating tar ball for log dir ${logdir}\n"
tarball="${USER}_$(date '+%d%m%Y_%H%M%S')_$(basename -- ${logdir}).tar"

tar cvf ${tarball} ${logdir}
echo -e "[Info] Created ${tarball}\n"

echo "[Info] Uploading ${tarball} ..."
time curl -u ${token}:${psw} -T ${tarball} "https://owncloud.tuwien.ac.at/public.php/webdav/${tarball}"
echo "[Info] Done."