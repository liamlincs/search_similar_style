App({
  onError(err) {
    console.error("[app:onError]", err);
  },
  onUnhandledRejection(res) {
    console.error("[app:onUnhandledRejection]", res);
  },
  onPageNotFound(res) {
    console.error("[app:onPageNotFound]", res);
  }
})
